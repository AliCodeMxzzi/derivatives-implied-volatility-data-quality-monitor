"""Daily data quality report (Markdown) and the metrics behind alerts.

Every number comes from a database query. A section whose tables are missing
(because an earlier step failed) says so instead of crashing the report.
"""
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from ivmon import config
from ivmon.svi import plot_smiles

log = logging.getLogger(__name__)


def _query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame | None:
    """Run a query; None if it fails (e.g. a table was never created)."""
    try:
        return con.execute(sql, params or []).df()
    except duckdb.Error as exc:
        log.warning("report query failed: %s", exc)
        return None


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    df = _query(con, sql, params)
    if df is None or df.empty:
        return None
    value = df.iloc[0, 0]
    return None if pd.isna(value) else value


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, pd.Timestamp):
        # expiries are dates (midnight); snapshots carry a time we must keep
        return value.strftime("%Y-%m-%d" if value == value.normalize() else "%Y-%m-%d %H:%M:%S")
    return str(value)


def md_table(df: pd.DataFrame | None) -> str:
    """A small Markdown table (no extra dependency)."""
    if df is None:
        return "_unavailable (an earlier step failed)_"
    if df.empty:
        return "_none_"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------- metrics

def collect_metrics(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
    ingest_rows: dict[str, int] | None, failed_steps: list[str] | None,
) -> dict:
    """The numbers the alert rules look at (None when they can't be computed)."""
    n_raw = _scalar(con, "SELECT count(*) FROM raw_quotes WHERE snapshot_ts = ?", [snapshot_ts])
    zero_bid = _scalar(
        con, "SELECT avg(CAST(zero_bid AS INTEGER)) FROM clean_quotes WHERE snapshot_ts = ?",
        [snapshot_ts])
    fits = _query(con, "SELECT fit_status, count(*) AS n FROM svi_params "
                       "WHERE snapshot_ts = ? GROUP BY 1", [snapshot_ts])
    calibration_failure_rate = None
    if fits is not None and not fits.empty:
        total = int(fits["n"].sum())
        # too_few_points is a deliberate skip, not a calibration failure
        failed = int(fits[~fits["fit_status"].isin(["ok", "too_few_points"])]["n"].sum())
        calibration_failure_rate = failed / total
    n_fitted = _scalar(con, "SELECT count(*) FROM svi_residuals WHERE snapshot_ts = ?", [snapshot_ts])
    flagged = _scalar(
        con,
        "SELECT count(*) FROM (SELECT DISTINCT underlying, expiry, strike, "
        "COALESCE(option_type, '*') FROM anomalies WHERE snapshot_ts = ? "
        "AND NOT is_synthetic AND rule <> 'ml_isolation_forest' AND strike IS NOT NULL)",
        [snapshot_ts])
    anomaly_rate = (flagged / n_fitted) if flagged is not None and n_fitted else None
    return {
        "snapshot_ts": snapshot_ts, "n_raw_quotes": n_raw,
        "ingest_rows": ingest_rows or {}, "zero_bid_rate": zero_bid,
        "calibration_failure_rate": calibration_failure_rate,
        "n_fitted_quotes": n_fitted, "n_flagged_quotes": flagged,
        "anomaly_rate": anomaly_rate, "failed_steps": failed_steps or [],
    }


# ------------------------------------------------------------------ sections

def per_underlying(con: duckdb.DuckDBPyConnection, ts: datetime) -> pd.DataFrame | None:
    sql = """
    WITH q AS (
        SELECT underlying, count(*) AS quotes,
               round(100 * avg(CAST(zero_bid AS INTEGER)), 1) AS zero_bid_pct,
               round(100 * avg(CAST(wide_spread AS INTEGER)), 1) AS wide_pct,
               round(100 * avg(CAST(stale_trade AS INTEGER)), 1) AS stale_pct,
               round(100 * avg(CAST(low_liquidity AS INTEGER)), 1) AS low_liq_pct,
               round(100 * avg(CAST(crossed_market AS INTEGER)), 1) AS crossed_pct
        FROM clean_quotes WHERE snapshot_ts = ? GROUP BY 1),
    i AS (
        SELECT underlying, count(*) AS otm, count(own_iv) AS iv_solved
        FROM iv_quotes WHERE snapshot_ts = ? GROUP BY 1),
    f AS (
        SELECT underlying, count(*) FILTER (WHERE fit_status = 'ok') AS fits_ok,
               count(*) AS expiries,
               round(100 * median(rmse) FILTER (WHERE fit_status = 'ok'), 2) AS median_rmse_vp
        FROM svi_params WHERE snapshot_ts = ? GROUP BY 1),
    a AS (
        SELECT underlying, count(*) AS findings FROM anomalies
        WHERE snapshot_ts = ? AND NOT is_synthetic GROUP BY 1)
    SELECT q.*, i.otm, i.iv_solved, f.fits_ok, f.expiries, f.median_rmse_vp,
           COALESCE(a.findings, 0) AS findings
    FROM q LEFT JOIN i USING (underlying) LEFT JOIN f USING (underlying)
    LEFT JOIN a USING (underlying) ORDER BY underlying
    """
    return _query(con, sql, [ts, ts, ts, ts])


def findings_by_rule(con: duckdb.DuckDBPyConnection, ts: datetime) -> pd.DataFrame | None:
    return _query(
        con, "SELECT rule, severity, count(*) AS n FROM anomalies WHERE snapshot_ts = ? "
             "AND NOT is_synthetic GROUP BY 1, 2 ORDER BY 1, "
             "CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END", [ts])


def top_anomalies(con: duckdb.DuckDBPyConnection, ts: datetime, n: int) -> pd.DataFrame | None:
    return _query(
        con,
        "SELECT underlying, expiry, strike, option_type AS type, rule, severity, details "
        "FROM anomalies WHERE snapshot_ts = ? AND NOT is_synthetic "
        "ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
        "rule, underlying, expiry, strike LIMIT ?", [ts, n])


def remediation_summary(con: duckdb.DuckDBPyConnection, ts: datetime) -> pd.DataFrame | None:
    return _query(
        con,
        "SELECT rule, action, count(*) AS n FROM remediation_log WHERE snapshot_ts = ? "
        "AND NOT is_synthetic AND run_id = (SELECT max(run_id) FROM remediation_runs "
        "WHERE snapshot_ts = ? AND NOT is_synthetic) GROUP BY 1, 2 ORDER BY 1, 2", [ts, ts])


def trend(con: duckdb.DuckDBPyConnection, n: int) -> pd.DataFrame | None:
    """Recent snapshots: clean-quote rates for all, findings/fits where processed."""
    return _query(
        con,
        """
        WITH snaps AS (
            SELECT snapshot_ts, count(*) AS quotes,
                   round(100 * avg(CAST(zero_bid AS INTEGER)), 1) AS zero_bid_pct,
                   round(100 * avg(CAST(wide_spread AS INTEGER)), 1) AS wide_pct,
                   round(100 * avg(CAST(stale_trade AS INTEGER)), 1) AS stale_pct
            FROM clean_quotes GROUP BY 1 ORDER BY 1 DESC LIMIT ?)
        SELECT s.*,
               (SELECT count(*) FROM anomalies a WHERE a.snapshot_ts = s.snapshot_ts
                  AND NOT a.is_synthetic) AS findings,
               (SELECT count(*) FROM svi_params p WHERE p.snapshot_ts = s.snapshot_ts
                  AND p.fit_status = 'ok') AS fits_ok
        FROM snaps s ORDER BY s.snapshot_ts DESC
        """, [n])


# -------------------------------------------------------------------- report

def write_report(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, metrics: dict,
    steps: list, alerts: list, refit: dict | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Write the Markdown report (and surface plots). Returns the file path."""
    out_dir = Path(out_dir if out_dir is not None else config.REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(snapshot_ts).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"report_{stamp}.md"
    plots_dir = out_dir / f"report_{stamp}_plots"

    status = pd.DataFrame([{"step": s.name, "status": s.status,
                            "seconds": round(s.seconds, 1), "detail": s.detail}
                           for s in steps])
    alert_lines = ([f"- **{a.severity.upper()}** `{a.name}`: {a.message}" for a in alerts]
                   or ["_No alerts._"])

    parts = [
        f"# IV Data Quality Report: snapshot {snapshot_ts} UTC\n",
        "## Run status\n", md_table(status), "\n## Alerts\n", "\n".join(alert_lines),
        "\n## Per-underlying summary\n",
        "_quotes = raw rows; flag columns are % of quotes; otm / iv_solved = OTM quotes and "
        "how many got our own IV; median_rmse_vp = median SVI fit RMSE in vol points._\n",
        md_table(per_underlying(con, snapshot_ts)),
        "\n## Findings by rule and severity\n", md_table(findings_by_rule(con, snapshot_ts)),
        f"\n## Top {config.REPORT_TOP_N} anomalies (high severity first)\n",
        md_table(top_anomalies(con, snapshot_ts, config.REPORT_TOP_N)),
        "\n## Remediation (latest run)\n", md_table(remediation_summary(con, snapshot_ts)),
    ]
    if refit:
        parts.append(
            f"\nRefit check on {refit['n_quotes']} untouched quotes in {refit['n_expiries']} "
            f"expiries: RMSE {refit['rmse_before']:.3f} -> {refit['rmse_after']:.3f} vol points "
            "(lower is better).")
    parts += ["\n## Trend vs prior snapshots\n",
              "_Clean-quote rates exist for every ingested snapshot; findings and fits only for "
              "snapshots the pipeline processed._\n",
              md_table(trend(con, config.TREND_SNAPSHOTS))]

    parts.append("\n## Volatility surface fits (market IV vs SVI)\n")
    underlyings = _query(con, "SELECT DISTINCT underlying FROM svi_params WHERE snapshot_ts = ? "
                              "ORDER BY 1", [snapshot_ts])
    plotted = False
    if underlyings is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for u in underlyings["underlying"]:
            try:
                image = plot_smiles(con, snapshot_ts, u, n_expiries=3,
                                    path=plots_dir / f"svi_{u}.png")
            except Exception as exc:  # a plotting problem must not lose the report
                log.warning("plot failed for %s: %s", u, exc)
                image = None
            if image:
                parts.append(f"**{u}**\n\n![{u} smiles]({plots_dir.name}/svi_{u}.png)\n")
                plotted = True
    if not plotted:
        parts.append("_No fitted surfaces to plot._")

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    log.info("report written to %s", path)
    return path
