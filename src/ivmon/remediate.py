"""Remediation and audit trail.

Findings -> actions (policy in config) -> append-only remediation_log ->
curated_quotes view. Raw data is never modified: the view applies the
actions on top of clean_quotes.

Precedence when one quote has several actions: quarantine > correct >
escalate for the "money" action (never publish a replacement for a quote
another rule says to exclude); an escalate is logged in addition.
"""
import logging
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from ivmon import config
from ivmon.db import get_connection, init_schema
from ivmon.pricing import black_price
from ivmon.svi import fit_expiries, svi_total_variance, _replace_rows

log = logging.getLogger(__name__)

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
ACTION_PRIORITY = {"quarantine": 3, "correct": 2, "escalate": 1}
QUOTE_KEY = ["underlying", "expiry", "strike", "option_type"]
FINDING_COLS = ["underlying", "expiry", "strike", "option_type", "rule", "severity", "details"]

LOG_COLUMNS = [
    "run_id", "run_ts", "snapshot_ts", "underlying", "expiry", "strike",
    "option_type", "action", "rule", "severity", "reason", "before_bid",
    "before_ask", "before_mid", "after_mid", "is_synthetic",
]


# -------------------------------------------------------------------- policy

def policy_action(rule: str, severity: str) -> str:
    """Policy lookup. Unknown rules fall back to the safest action."""
    by_severity = config.REMEDIATION_POLICY.get(rule)
    if by_severity is None:
        return config.REMEDIATION_DEFAULT_ACTION
    return by_severity.get(severity, config.REMEDIATION_DEFAULT_ACTION)


def load_findings(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, synthetic: bool,
) -> pd.DataFrame:
    """Anomaly findings plus crossed-market quotes (a Stage 2 flag) for a snapshot."""
    anomalies = con.execute(
        "SELECT underlying, expiry, strike, option_type, rule, severity, details "
        "FROM anomalies WHERE snapshot_ts = ? AND is_synthetic = ?",
        [snapshot_ts, synthetic]).df()
    crossed = con.execute(
        "SELECT underlying, expiry, strike, option_type, 'crossed_market' AS rule, "
        "'high' AS severity, 'bid ' || bid || ' > ask ' || ask AS details "
        "FROM clean_quotes WHERE snapshot_ts = ? AND crossed_market",
        [snapshot_ts]).df()
    frames = [f for f in (anomalies, crossed) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FINDING_COLS)


def decide_actions(findings: pd.DataFrame) -> pd.DataFrame:
    """Apply the policy; return one row per (quote or scope, action).

    Columns: QUOTE_KEY, action, rule, severity, reason. 'none' actions are dropped.
    """
    rows = []
    for f in findings.itertuples():
        action = policy_action(f.rule, f.severity)
        if action == "none":
            continue
        rows.append({
            "underlying": f.underlying, "expiry": f.expiry, "strike": f.strike,
            "option_type": f.option_type, "action": action, "rule": f.rule,
            "severity": f.severity,
            "reason": f"{f.details} [policy: {f.rule}/{f.severity} -> {action}]",
        })
    if not rows:
        return pd.DataFrame(columns=QUOTE_KEY + ["action", "rule", "severity", "reason"])
    acts = pd.DataFrame(rows)

    # Keys with NULLs don't group reliably, so use an explicit string key.
    acts["_key"] = [
        "|".join(str(v) for v in row) for row in acts[QUOTE_KEY].itertuples(index=False)]
    resolved = []
    for _, g in acts.groupby("_key", sort=False):
        money = g[g["action"].isin(["quarantine", "correct"])]
        escalate = g[g["action"] == "escalate"]
        groups = []  # (action, triggering findings)
        if not money.empty:
            # The strongest money action wins, but the log keeps EVERY trigger
            # (including a superseded 'correct') so the trail shows why.
            groups.append((max(money["action"], key=ACTION_PRIORITY.get), money))
        if not escalate.empty:
            groups.append(("escalate", escalate))
        for action, sub in groups:
            resolved.append({
                **sub.iloc[0][QUOTE_KEY].to_dict(), "action": action,
                "rule": "; ".join(sorted(set(sub["rule"]))),
                "severity": max(sub["severity"], key=SEVERITY_RANK.get),
                "reason": " | ".join(sub["reason"]),
            })
    return pd.DataFrame(resolved)


# ----------------------------------------------------------------- applying

def _before_after(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, actions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach before-values (from clean_quotes) and, for 'correct', the surface value."""
    quotes = con.execute(
        "SELECT underlying, expiry, strike, option_type, bid AS before_bid, "
        "ask AS before_ask, mid AS before_mid FROM clean_quotes WHERE snapshot_ts = ?",
        [snapshot_ts]).df()
    fitted = con.execute(
        "SELECT r.underlying, r.expiry, r.strike, r.option_type, r.fit_iv, "
        "i.forward, i.tte_years FROM svi_residuals r JOIN iv_quotes i USING "
        "(snapshot_ts, underlying, expiry, option_type, strike) WHERE r.snapshot_ts = ?",
        [snapshot_ts]).df()
    out = actions.merge(quotes, on=QUOTE_KEY, how="left").merge(fitted, on=QUOTE_KEY, how="left")
    after = []
    for r in out.itertuples():
        if r.action == "correct" and pd.notna(r.fit_iv):
            after.append(black_price(r.forward, r.strike, r.tte_years, r.fit_iv,
                                     config.RISK_FREE_RATE, r.option_type))
        else:
            after.append(np.nan)
    out["after_mid"] = after
    # A correction needs a surface value; without one, fall back to quarantine.
    no_value = (out["action"] == "correct") & out["after_mid"].isna()
    out.loc[no_value, "action"] = "quarantine"
    out.loc[no_value, "reason"] += " [no surface value available: quarantined instead]"
    return out


def apply_remediation(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime | None = None,
    synthetic: bool = False,
) -> pd.DataFrame:
    """Decide actions for a snapshot and APPEND them to remediation_log.

    Returns the rows written. Each call is a new run_id; nothing is updated
    or deleted. Also (re)creates the curated_quotes view.
    """
    if snapshot_ts is None:
        snapshot_ts = con.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]
    findings = load_findings(con, snapshot_ts, synthetic)
    actions = decide_actions(findings)

    run_id = con.execute("SELECT COALESCE(max(run_id), 0) + 1 FROM remediation_runs").fetchall()[0][0]
    run_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    rows = pd.DataFrame(columns=LOG_COLUMNS)
    if not actions.empty:
        rows = _before_after(con, snapshot_ts, actions)
        rows["run_id"] = run_id
        rows["run_ts"] = run_ts
        rows["snapshot_ts"] = snapshot_ts
        rows["is_synthetic"] = synthetic
        rows = rows[LOG_COLUMNS].copy()
        for col in ["strike", "before_bid", "before_ask", "before_mid", "after_mid"]:
            rows[col] = rows[col].astype("Float64")
        con.register("log_in", rows)
        try:
            cols = ", ".join(LOG_COLUMNS)
            con.execute(f"INSERT INTO remediation_log ({cols}) SELECT {cols} FROM log_in").fetchall()
        finally:
            con.unregister("log_in")
    # Record the run itself, even if it took no actions.
    con.execute("INSERT INTO remediation_runs VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, run_ts, snapshot_ts, synthetic, len(findings), len(rows)]).fetchall()
    create_curated_view(con)
    log.info("remediation run %d: %d actions from %d findings", run_id, len(rows), len(findings))
    return rows


CURATED_VIEW_SQL = """
CREATE OR REPLACE VIEW curated_quotes AS
WITH latest AS (
    SELECT snapshot_ts, is_synthetic, max(run_id) AS run_id
    FROM remediation_runs GROUP BY snapshot_ts, is_synthetic
),
acts AS (
    SELECT l.* FROM remediation_log l
    JOIN latest USING (snapshot_ts, is_synthetic, run_id)
    WHERE l.strike IS NOT NULL AND l.option_type IS NOT NULL
),
per_quote AS (
    SELECT snapshot_ts, underlying, expiry, option_type, strike,
           bool_or(action = 'quarantine') AS is_quarantined,
           bool_or(action = 'correct') AS is_corrected,
           bool_or(action = 'escalate') AS is_escalated,
           max(after_mid) FILTER (WHERE action = 'correct') AS corrected_mid,
           string_agg(action || ':' || rule, '; ') AS applied
    FROM acts GROUP BY ALL
)
SELECT c.snapshot_ts, c.underlying, c.expiry, c.option_type, c.strike,
       c.bid, c.ask, c.mid AS raw_mid,
       CASE WHEN COALESCE(p.is_corrected, FALSE) AND NOT COALESCE(p.is_quarantined, FALSE)
            THEN p.corrected_mid ELSE c.mid END AS curated_mid,
       COALESCE(p.is_quarantined, FALSE) AS is_quarantined,
       COALESCE(p.is_corrected, FALSE) AS is_corrected,
       COALESCE(p.is_escalated, FALSE) AS is_escalated,
       p.applied AS actions_applied
FROM clean_quotes c
LEFT JOIN per_quote p USING (snapshot_ts, underlying, expiry, option_type, strike)
"""


def create_curated_view(con: duckdb.DuckDBPyConnection) -> None:
    """curated_quotes = clean_quotes + the latest run's quote-level actions."""
    con.execute(CURATED_VIEW_SQL).fetchall()


# -------------------------------------------------------------------- refit

def _acted_keys(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
                synthetic: bool) -> set[tuple]:
    """Quotes quarantined or corrected in the latest run (excluded from the refit)."""
    rows = con.execute(
        """
        SELECT underlying, expiry, option_type, strike FROM remediation_log
        WHERE snapshot_ts = ? AND is_synthetic = ? AND action IN ('quarantine', 'correct')
          AND strike IS NOT NULL AND option_type IS NOT NULL
          AND run_id = (SELECT max(run_id) FROM remediation_runs
                        WHERE snapshot_ts = ? AND is_synthetic = ?)
        """, [snapshot_ts, synthetic, snapshot_ts, synthetic]).fetchall()
    return {(u, pd.Timestamp(e), t, float(k)) for u, e, t, k in rows}


def _solved_quotes(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM iv_quotes WHERE snapshot_ts = ? AND own_iv IS NOT NULL "
        "AND tte_years > 0 AND forward > 0", [snapshot_ts]).df()


def _mark_acted(quotes: pd.DataFrame, acted: set[tuple]) -> pd.Series:
    keys = zip(quotes["underlying"], pd.to_datetime(quotes["expiry"]),
               quotes["option_type"], quotes["strike"].astype(float))
    return pd.Series([k in acted for k in keys], index=quotes.index)


def refit_curated(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, synthetic: bool = False,
) -> pd.DataFrame:
    """Refit SVI without quarantined/corrected quotes -> svi_params_curated.

    Corrected quotes are excluded too: their value came from the old fit, so
    feeding them back would let the model confirm itself.
    """
    quotes = _solved_quotes(con, snapshot_ts)
    keep = quotes[~_mark_acted(quotes, _acted_keys(con, snapshot_ts, synthetic))]
    param_rows, _ = fit_expiries(keep, snapshot_ts)
    params = pd.DataFrame(param_rows)
    _replace_rows(con, "svi_params_curated", snapshot_ts, params)
    return params


def _iv_errors(params: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Per-quote (market IV - curve IV) using each expiry's fitted curve."""
    ok = params[params["fit_status"] == "ok"][["underlying", "expiry", "a", "b", "rho", "m", "sigma"]]
    q = quotes.merge(ok, on=["underlying", "expiry"], how="inner")
    k = np.log(q["strike"].to_numpy(float) / q["forward"].to_numpy(float))
    w = svi_total_variance(k, q["a"].to_numpy(float), q["b"].to_numpy(float),
                           q["rho"].to_numpy(float), q["m"].to_numpy(float),
                           q["sigma"].to_numpy(float))
    q["err"] = q["own_iv"].to_numpy(float) - np.sqrt(np.maximum(w, 0) / q["tte_years"].to_numpy(float))
    return q[["underlying", "expiry", "option_type", "strike", "err"]]


def compare_fit(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, synthetic: bool = False,
    exclude_keys: set[tuple] | None = None,
) -> dict:
    """Fit quality before vs after remediation, on ONE common set of quotes.

    Common set = solved quotes that remediation left alone (and, if given,
    that are not in exclude_keys, e.g. the injected ones), in expiries whose
    fit succeeded both before and after. Both curves are scored on exactly
    these quotes, so the comparison is like-for-like.
    """
    pre = con.execute("SELECT * FROM svi_params WHERE snapshot_ts = ?", [snapshot_ts]).df()
    post = con.execute("SELECT * FROM svi_params_curated WHERE snapshot_ts = ?", [snapshot_ts]).df()
    quotes = _solved_quotes(con, snapshot_ts)
    drop = _mark_acted(quotes, _acted_keys(con, snapshot_ts, synthetic))
    if exclude_keys:
        drop |= _mark_acted(quotes, exclude_keys)
    common = quotes[~drop]

    both_ok = set(map(tuple, pre[pre.fit_status == "ok"][["underlying", "expiry"]].to_numpy())) & \
        set(map(tuple, post[post.fit_status == "ok"][["underlying", "expiry"]].to_numpy()))
    common = common[[(u, e) in both_ok for u, e in zip(common["underlying"], common["expiry"])]]

    e_pre = _iv_errors(pre, common).rename(columns={"err": "err_pre"})
    e_post = _iv_errors(post, common).rename(columns={"err": "err_post"})
    errs = e_pre.merge(e_post, on=["underlying", "expiry", "option_type", "strike"])

    def rmse(x: pd.Series) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) * 100  # vol points

    by_underlying = errs.groupby("underlying").agg(
        n_quotes=("err_pre", "size"),
        rmse_before=("err_pre", rmse), rmse_after=("err_post", rmse)).round(3).reset_index()
    overall = {
        "n_quotes": len(errs),
        "n_expiries": len(both_ok),
        "rmse_before": rmse(errs["err_pre"]) if len(errs) else float("nan"),
        "rmse_after": rmse(errs["err_post"]) if len(errs) else float("nan"),
    }
    return {"by_underlying": by_underlying, "overall": overall}


def action_summary(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
                   synthetic: bool = False) -> pd.DataFrame:
    return con.execute(
        "SELECT rule, action, count(*) AS n FROM remediation_log WHERE snapshot_ts = ? "
        "AND is_synthetic = ? AND run_id = (SELECT max(run_id) FROM remediation_runs "
        "WHERE snapshot_ts = ? AND is_synthetic = ?) GROUP BY 1, 2 ORDER BY 1, 2",
        [snapshot_ts, synthetic, snapshot_ts, synthetic]).df()


def main() -> None:
    from ivmon import synthetic as syn_mod

    logging.basicConfig(level=logging.WARNING)
    pd.set_option("display.width", 200)
    real = get_connection(config.DB_PATH)
    init_schema(real)
    ts = real.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]

    print(f"\n=== REAL snapshot {ts} ===")
    apply_remediation(real, ts)
    print("\nActions taken (latest run):")
    print(action_summary(real, ts).to_string(index=False))
    n_cur = real.execute(
        "SELECT count(*) FILTER (WHERE is_quarantined), count(*) FILTER (WHERE is_corrected), "
        "count(*) FILTER (WHERE is_escalated), count(*) FROM curated_quotes "
        "WHERE snapshot_ts = ?", [ts]).fetchall()[0]
    print(f"\ncurated_quotes: {n_cur[3]} rows; quarantined {n_cur[0]}, corrected {n_cur[1]}, "
          f"escalated {n_cur[2]}")
    refit_curated(real, ts)
    cmp_real = compare_fit(real, ts)
    print("\nFit quality, scored on the SAME untouched quotes (vol points):")
    print(cmp_real["by_underlying"].to_string(index=False))
    o = cmp_real["overall"]
    print(f"overall: {o['n_quotes']} quotes in {o['n_expiries']} expiries: "
          f"RMSE {o['rmse_before']:.3f} -> {o['rmse_after']:.3f}")

    print("\n=== SYNTHETIC copy (corruptions injected, truth known) ===")
    syn, truth = syn_mod.build_synthetic(real, ts)
    apply_remediation(syn, ts, synthetic=True)
    print("\nActions taken (latest run):")
    print(action_summary(syn, ts, synthetic=True).to_string(index=False))
    refit_curated(syn, ts, synthetic=True)
    injected = {(r.underlying, pd.Timestamp(r.expiry), r.option_type, float(r.strike))
                for r in truth.itertuples()}
    cmp_syn = compare_fit(syn, ts, synthetic=True, exclude_keys=injected)
    o = cmp_syn["overall"]
    print("\nFit quality scored on quotes that were NEVER injected and not acted on (vol points):")
    print(cmp_syn["by_underlying"].to_string(index=False))
    print(f"overall: {o['n_quotes']} quotes in {o['n_expiries']} expiries: "
          f"RMSE {o['rmse_before']:.3f} -> {o['rmse_after']:.3f}")


if __name__ == "__main__":
    main()
