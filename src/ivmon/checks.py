"""Rule-based anomaly checks: no-arbitrage and consistency.

Each check takes plain DataFrames and returns a DataFrame of findings with
FINDING_COLUMNS, so every rule is testable with small hand-built inputs.
`run_checks` reads the database, runs all rules, and stores the findings.

Rules:  butterfly_arbitrage, calendar_arbitrage, put_call_parity,
        residual_outlier, param_stability
"""
import logging
import math
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from ivmon import config
from ivmon.db import get_connection, init_schema
from ivmon.svi import svi_total_variance

log = logging.getLogger(__name__)

FINDING_COLUMNS = [
    "snapshot_ts", "underlying", "expiry", "strike", "option_type",
    "rule", "severity", "details",
]
SVI_COLS = ["a", "b", "rho", "m", "sigma"]


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FINDING_COLUMNS)


def _finding(p, rule: str, severity: str, details: str,
             strike: float | None = None, option_type: str | None = None) -> dict:
    """One finding row. `p` is any row with snapshot_ts/underlying/expiry."""
    return {
        "snapshot_ts": p["snapshot_ts"], "underlying": p["underlying"],
        "expiry": p["expiry"], "strike": strike, "option_type": option_type,
        "rule": rule, "severity": severity, "details": details,
    }


# ---------------------------------------------------------------- 1. butterfly

def svi_density_g(k: np.ndarray, a: float, b: float, rho: float, m: float,
                  sigma: float) -> np.ndarray:
    """Gatheral's butterfly function g(k) for raw SVI.

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2.
    The implied probability density is proportional to g, so g < 0 means
    negative density, i.e. a butterfly arbitrage.
    """
    k = np.asarray(k, dtype=float)
    root = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = a + b * (rho * (k - m) + root)
    w1 = b * (rho + (k - m) / root)
    w2 = b * sigma ** 2 / root ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def butterfly_check(params: pd.DataFrame, resid: pd.DataFrame) -> pd.DataFrame:
    """Flag quotes whose log-moneyness falls where the fitted g(k) < 0.

    If the violation lies only between quotes, emit one expiry-level finding.
    """
    rows = []
    for _, p in params.iterrows():
        pts = resid[(resid["underlying"] == p["underlying"])
                    & (resid["expiry"] == p["expiry"])]
        if pts.empty:
            continue
        args = tuple(float(p[c]) for c in SVI_COLS)
        g_quotes = svi_density_g(pts["log_moneyness"].to_numpy(), *args)
        bad = g_quotes < -config.BUTTERFLY_TOL
        for (_, q), g in zip(pts[bad].iterrows(), g_quotes[bad]):
            sev = "high" if g < config.BUTTERFLY_HIGH else "medium"
            rows.append(_finding(
                p, "butterfly_arbitrage", sev,
                f"g(k)={g:.3f} < 0 at k={q['log_moneyness']:.3f} (negative density)",
                q["strike"], q["option_type"]))
        if not bad.any():
            grid = np.linspace(pts["log_moneyness"].min(), pts["log_moneyness"].max(), 400)
            g_grid = svi_density_g(grid, *args)
            if g_grid.min() < -config.BUTTERFLY_TOL:
                k_bad = grid[g_grid.argmin()]
                sev = "high" if g_grid.min() < config.BUTTERFLY_HIGH else "medium"
                rows.append(_finding(
                    p, "butterfly_arbitrage", sev,
                    f"min g(k)={g_grid.min():.3f} at k={k_bad:.3f}, between quotes"))
    return _frame(rows)


# ----------------------------------------------------------------- 2. calendar

def calendar_check(params: pd.DataFrame, resid: pd.DataFrame) -> pd.DataFrame:
    """Total variance must not decrease with expiry at fixed log-moneyness.

    For consecutive expiries of one underlying, evaluate both fitted curves at
    each quote's k in the later expiry (only where k lies inside the earlier
    expiry's fitted range, to avoid extrapolating) and flag drops.
    """
    rows = []
    for _, fits in params.groupby("underlying"):
        fits = fits.sort_values("expiry").reset_index(drop=True)
        for i in range(1, len(fits)):
            early, late = fits.iloc[i - 1], fits.iloc[i]
            e_pts = resid[(resid["underlying"] == early["underlying"])
                          & (resid["expiry"] == early["expiry"])]
            l_pts = resid[(resid["underlying"] == late["underlying"])
                          & (resid["expiry"] == late["expiry"])]
            if e_pts.empty or l_pts.empty:
                continue
            lo, hi = e_pts["log_moneyness"].min(), e_pts["log_moneyness"].max()
            inside = l_pts[(l_pts["log_moneyness"] >= lo) & (l_pts["log_moneyness"] <= hi)]
            k = inside["log_moneyness"].to_numpy()
            w_early = svi_total_variance(k, *(float(early[c]) for c in SVI_COLS))
            w_late = svi_total_variance(k, *(float(late[c]) for c in SVI_COLS))
            drop = (w_early - w_late) / w_early
            for (_, q), d, we, wl in zip(inside.iterrows(), drop, w_early, w_late):
                if d <= config.CALENDAR_TOL_REL:
                    continue
                if d >= config.CALENDAR_HIGH_REL:
                    sev = "high"
                elif d >= config.CALENDAR_MEDIUM_REL:
                    sev = "medium"
                else:
                    sev = "low"
                rows.append(_finding(
                    late, "calendar_arbitrage", sev,
                    f"w at k={q['log_moneyness']:.3f} falls {d:.1%} vs earlier expiry "
                    f"{pd.Timestamp(early['expiry']).date()} ({we:.5f} -> {wl:.5f})",
                    q["strike"], q["option_type"]))
    return _frame(rows)


# ------------------------------------------------------------------- 3. parity

def parity_check(quotes: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Put-call parity at each strike vs the expiry's estimated forward.

    Parity: C - P = D (F - K). The gap should stay inside the quoted spreads,
    so tolerance = PARITY_TOL_MULT * average(call spread, put spread), floored.
    Only expiries whose forward came from parity are checked (a spot-based
    fallback forward would make every strike look broken).
    Parity is exact only for European options: we stay near ATM to limit the
    early-exercise effect.
    """
    rows = []
    ok_fwd = forwards[forwards["forward_method"] == "parity"]
    for _, f in ok_fwd.iterrows():
        q = quotes[(quotes["underlying"] == f["underlying"])
                   & (quotes["expiry"] == f["expiry"])
                   & ~quotes["crossed_market"].astype(bool)
                   & ~quotes["zero_bid"].astype(bool)
                   & quotes["mid"].notna()]
        calls = q[q["option_type"] == "C"].set_index("strike")
        puts = q[q["option_type"] == "P"].set_index("strike")
        discount = math.exp(-config.RISK_FREE_RATE * float(f["tte_years"]))
        for strike in calls.index.intersection(puts.index):
            if abs(math.log(strike / f["forward"])) > config.PARITY_MAX_LOG_MONEYNESS:
                continue
            c, p = calls.loc[strike], puts.loc[strike]
            gap = (c["mid"] - p["mid"]) - discount * (f["forward"] - strike)
            tol = max(config.PARITY_TOL_MULT * (c["spread"] + p["spread"]) / 2,
                      config.PARITY_MIN_ABS)
            if abs(gap) <= tol:
                continue
            ratio = abs(gap) / tol
            if ratio >= config.PARITY_HIGH_RATIO:
                sev = "high"
            elif ratio >= config.PARITY_MEDIUM_RATIO:
                sev = "medium"
            else:
                sev = "low"
            rows.append(_finding(
                f, "put_call_parity", sev,
                f"C-P-D(F-K) = {gap:+.3f} vs tolerance {tol:.3f} "
                f"(call mid {c['mid']:.2f}, put mid {p['mid']:.2f})",
                float(strike), None))  # pair-level: option_type NULL
    return _frame(rows)


# ----------------------------------------------------------------- 4. residual

def residual_outlier_check(resid: pd.DataFrame) -> pd.DataFrame:
    """Flag quotes far from their expiry's SVI fit, using a robust z-score.

    z = 0.6745 * (r - median) / MAD. Mean/std would be inflated by the very
    outliers we want to catch; median/MAD are not. A quote must also miss by
    at least RESID_MIN_ABS_IV so a near-perfect fit (tiny MAD) doesn't make
    trivial residuals look extreme.
    """
    rows = []
    for _, g in resid.groupby(["underlying", "expiry"]):
        r = g["resid_iv"].to_numpy(dtype=float)
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad == 0:
            continue  # cannot scale; skip rather than divide by zero
        z = 0.6745 * (r - med) / mad
        for (_, q), zi, ri in zip(g.iterrows(), z, r):
            if abs(zi) < config.RESID_Z_THRESHOLD or abs(ri) < config.RESID_MIN_ABS_IV:
                continue
            if abs(zi) >= config.RESID_HIGH_Z:
                sev = "high"
            elif abs(zi) >= config.RESID_MEDIUM_Z:
                sev = "medium"
            else:
                sev = "low"
            rows.append(_finding(
                q, "residual_outlier", sev,
                f"IV {q['own_iv']:.3f} vs fit {q['fit_iv']:.3f} "
                f"(resid {ri * 100:+.2f} vol pts, robust z={zi:+.1f})",
                q["strike"], q["option_type"]))
    return _frame(rows)


# ---------------------------------------------------------------- 5. stability

def _smile_iv(p, t: float, grid: np.ndarray) -> np.ndarray:
    w = svi_total_variance(grid, *(float(p[c]) for c in SVI_COLS))
    return np.sqrt(np.maximum(w, 0.0) / t)


def stability_check(
    current: pd.DataFrame, previous: pd.DataFrame,
    t_current: dict, t_previous: dict,
) -> pd.DataFrame:
    """Compare each expiry's smile with the previous snapshot's.

    SVI parameters are partly interchangeable (different a, b, m, sigma can
    draw nearly the same curve), so we compare the IV curves on a fixed
    log-moneyness grid, not the raw parameters. Raw parameter changes are
    still shown in the details. t_* map (underlying, expiry) -> years.
    """
    rows = []
    grid = np.linspace(-config.STABILITY_GRID_K, config.STABILITY_GRID_K, 41)
    prev = previous.set_index(["underlying", "expiry"])
    for _, p in current.iterrows():
        key = (p["underlying"], p["expiry"])
        if key not in prev.index or key not in t_current or key not in t_previous:
            continue
        q = prev.loc[key]
        iv_now = _smile_iv(p, t_current[key], grid)
        iv_before = _smile_iv(q, t_previous[key], grid)
        rms = float(np.sqrt(np.mean((iv_now - iv_before) ** 2)))
        if rms < config.STABILITY_RMS_IV:
            continue
        sev = "high" if rms >= config.STABILITY_HIGH_RMS_IV else "medium"
        deltas = ", ".join(f"{c} {q[c]:.3f}->{p[c]:.3f}" for c in SVI_COLS)
        rows.append(_finding(
            p, "param_stability", sev,
            f"smile moved {rms * 100:.1f} vol pts RMS vs previous snapshot ({deltas})"))
    return _frame(rows)


# ---------------------------------------------------------------- orchestration

def _previous_snapshot(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime):
    return con.execute(
        "SELECT max(snapshot_ts) FROM svi_params WHERE snapshot_ts < ?",
        [snapshot_ts]).fetchall()[0][0]


def run_checks(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime | None = None,
) -> pd.DataFrame:
    """Run all rules for one snapshot, store findings, return them."""
    if snapshot_ts is None:
        snapshot_ts = con.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]
    if snapshot_ts is None:
        log.warning("no SVI fits; nothing to check")
        return _frame([])

    params = con.execute(
        "SELECT * FROM svi_params WHERE snapshot_ts = ? AND fit_status = 'ok'",
        [snapshot_ts]).df()
    resid = con.execute(
        "SELECT * FROM svi_residuals WHERE snapshot_ts = ?", [snapshot_ts]).df()
    quotes = con.execute(
        "SELECT * FROM clean_quotes WHERE snapshot_ts = ?", [snapshot_ts]).df()
    forwards = con.execute(
        "SELECT DISTINCT snapshot_ts, underlying, expiry, forward, forward_method, "
        "tte_years FROM iv_quotes WHERE snapshot_ts = ?", [snapshot_ts]).df()

    parts = [
        butterfly_check(params, resid),
        calendar_check(params, resid),
        parity_check(quotes, forwards),
        residual_outlier_check(resid),
    ]

    prev_ts = _previous_snapshot(con, snapshot_ts)
    if prev_ts is None:
        log.info("param_stability SKIPPED: no previous snapshot with SVI fits")
    else:
        prev_params = con.execute(
            "SELECT * FROM svi_params WHERE snapshot_ts = ? AND fit_status = 'ok'",
            [prev_ts]).df()
        t_now = _tte_map(con, snapshot_ts)
        t_before = _tte_map(con, prev_ts)
        parts.append(stability_check(params, prev_params, t_now, t_before))

    findings = pd.concat(parts, ignore_index=True) if parts else _frame([])
    _store(con, snapshot_ts, findings)
    log.info("anomalies stored for %s: %d findings", snapshot_ts, len(findings))
    return findings


def _tte_map(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> dict:
    df = con.execute(
        "SELECT DISTINCT underlying, expiry, tte_years FROM svi_residuals "
        "WHERE snapshot_ts = ?", [snapshot_ts]).df()
    return {(r.underlying, r.expiry): float(r.tte_years) for r in df.itertuples()}


def _store(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
           findings: pd.DataFrame) -> None:
    """Replace this snapshot's real rule-based findings.

    Only the rules owned by this module are replaced; synthetic rows and ML
    findings (a different module) are left alone.
    """
    con.execute("DELETE FROM anomalies WHERE snapshot_ts = ? AND NOT is_synthetic "
                "AND rule <> 'ml_isolation_forest'", [snapshot_ts]).fetchall()
    if findings.empty:
        return
    out = findings.copy()
    out["strike"] = out["strike"].astype("Float64")
    out["is_synthetic"] = False
    con.register("incoming_findings", out)
    try:
        cols = ", ".join(list(FINDING_COLUMNS) + ["is_synthetic"])
        con.execute(f"INSERT INTO anomalies ({cols}) SELECT {cols} FROM incoming_findings").fetchall()
    finally:
        con.unregister("incoming_findings")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]
    run_checks(con, ts)
    pd.set_option("display.width", 200)
    print(f"\nSnapshot: {ts}\n\nFindings by rule and severity:")
    print(con.execute(
        "SELECT rule, severity, count(*) AS n FROM anomalies WHERE snapshot_ts = ? "
        "AND NOT is_synthetic GROUP BY 1, 2 ORDER BY 1, 2", [ts]).df().to_string(index=False))
    print("\nFindings by underlying:")
    print(con.execute(
        "SELECT underlying, count(*) AS n FROM anomalies WHERE snapshot_ts = ? "
        "AND NOT is_synthetic GROUP BY 1 ORDER BY 1", [ts]).df().to_string(index=False))


if __name__ == "__main__":
    main()
