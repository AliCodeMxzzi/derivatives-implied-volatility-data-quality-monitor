"""SVI (Stochastic Volatility Inspired) smile calibration.

Raw SVI for one expiry, in total variance w = sigma_iv^2 * T against
log-moneyness k = ln(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

a: overall variance level      b: slope/angle of the wings (>= 0)
rho: skew (tilt left/right)    m: horizontal shift of the smile's vertex
sigma: smoothness of the vertex (small = sharp V, large = rounded)
"""
import logging
import math
from datetime import datetime
from typing import NamedTuple

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ivmon import config
from ivmon.db import get_connection, init_schema

log = logging.getLogger(__name__)


class SVIFit(NamedTuple):
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    rmse: float  # unweighted, implied-vol units (decimal)
    n_points: int
    status: str  # ok / too_few_points / bad_input / negative_variance / fit_failed


def svi_total_variance(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float,
) -> np.ndarray:
    """Raw SVI total variance at log-moneyness k."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _failed(n: int, status: str) -> SVIFit:
    nan = float("nan")
    return SVIFit(nan, nan, nan, nan, nan, nan, n, status)


def _bounds(k: np.ndarray, w: np.ndarray) -> tuple[list[float], list[float]]:
    """Parameter bounds: a, b, rho, m, sigma."""
    w_max = float(w.max())
    lower = [-w_max, 1e-6, -0.999, float(k.min()) - 0.5, 1e-3]
    upper = [w_max, 10.0, 0.999, float(k.max()) + 0.5, 2.0]
    return lower, upper


def _initial_guesses(k: np.ndarray, w: np.ndarray) -> list[list[float]]:
    """A few starting points; SVI is non-convex so one start can get stuck."""
    lower, upper = _bounds(k, w)
    k_span = max(float(k.max() - k.min()), 0.1)
    m0 = float(k[np.argmin(w)])
    b0 = max(float(w.max() - w.min()) / k_span, 1e-3)
    guesses = []
    for rho0 in (-0.5, 0.0):
        for sigma0 in (0.05, 0.3):
            a0 = max(float(w.min()) * 0.5, lower[0])
            guess = [a0, b0, rho0, m0, sigma0]
            guesses.append([min(max(g, lo), hi)
                            for g, lo, hi in zip(guess, lower, upper)])
    return guesses


def fit_svi(
    k: np.ndarray, w: np.ndarray, t: float,
    weights: np.ndarray | None = None,
) -> SVIFit:
    """Fit raw SVI to (k, w) points for one expiry with time-to-expiry t.

    Weighted least squares with a robust (soft_l1) loss so one bad quote
    cannot drag the curve toward itself. Best of several starting points.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(k)
    if n < config.SVI_MIN_POINTS:
        return _failed(n, "too_few_points")
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(w))
            or np.any(w <= 0) or not t > 0):
        return _failed(n, "bad_input")

    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(weights / weights.mean())
    f_scale = max(config.SVI_ROBUST_FRACTION * float(np.median(w)), 1e-8)
    lower, upper = _bounds(k, w)

    def residuals(p: np.ndarray) -> np.ndarray:
        return sqrt_w * (svi_total_variance(k, *p) - w)

    best = None
    for x0 in _initial_guesses(k, w):
        try:
            sol = least_squares(residuals, x0, bounds=(lower, upper),
                                loss="soft_l1", f_scale=f_scale, max_nfev=2000)
        except (ValueError, FloatingPointError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return _failed(n, "fit_failed")

    a, b, rho, m, sigma = (float(x) for x in best.x)
    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
    if fitted_w.min() < 0:
        return SVIFit(a, b, rho, m, sigma, float("nan"), n, "negative_variance")
    fit_iv = np.sqrt(fitted_w / t)
    rmse = float(np.sqrt(np.mean((np.sqrt(w / t) - fit_iv) ** 2)))
    return SVIFit(a, b, rho, m, sigma, rmse, n, "ok")


def point_weights(rel_spread: pd.Series) -> np.ndarray:
    """Weight = 1 / relative spread (floored). Missing spread -> low weight."""
    spread = rel_spread.astype(float).fillna(1.0).clip(lower=config.SVI_MIN_REL_SPREAD)
    return (1.0 / spread).to_numpy()


def _load_solved_quotes(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM iv_quotes WHERE snapshot_ts = ? AND own_iv IS NOT NULL "
        "AND tte_years > 0 AND forward > 0", [snapshot_ts]).df()


def build_svi(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime | None = None,
) -> tuple[int, int]:
    """Fit every underlying/expiry in a snapshot. Returns (n_fits_ok, n_expiries)."""
    if snapshot_ts is None:
        snapshot_ts = con.execute("SELECT max(snapshot_ts) FROM iv_quotes").fetchall()[0][0]
    if snapshot_ts is None:
        log.warning("iv_quotes is empty; nothing to fit")
        return 0, 0

    quotes = _load_solved_quotes(con, snapshot_ts)
    param_rows, resid_frames = [], []
    for (underlying, expiry), g in quotes.groupby(["underlying", "expiry"]):
        t = float(g["tte_years"].iloc[0])
        k = np.log(g["strike"].to_numpy(dtype=float) / g["forward"].to_numpy(dtype=float))
        iv = g["own_iv"].to_numpy(dtype=float)
        w = iv ** 2 * t
        fit = fit_svi(k, w, t, point_weights(g["rel_spread"]))
        if fit.status != "ok":
            log.warning("%s %s: SVI skipped (%s, %d points)",
                        underlying, expiry, fit.status, fit.n_points)
        param_rows.append({
            "snapshot_ts": snapshot_ts, "underlying": underlying, "expiry": expiry,
            "a": fit.a, "b": fit.b, "rho": fit.rho, "m": fit.m, "sigma": fit.sigma,
            "rmse": fit.rmse, "n_points": fit.n_points, "fit_status": fit.status,
        })
        if fit.status != "ok":
            continue
        w_fit = svi_total_variance(k, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
        fit_iv = np.sqrt(np.maximum(w_fit, 0.0) / t)
        resid_frames.append(pd.DataFrame({
            "snapshot_ts": snapshot_ts, "underlying": underlying, "expiry": expiry,
            "option_type": g["option_type"].to_numpy(),
            "strike": g["strike"].to_numpy(dtype=float),
            "log_moneyness": k, "tte_years": t, "w_market": w, "w_fit": w_fit,
            "own_iv": iv, "fit_iv": fit_iv, "resid_w": w - w_fit,
            "resid_iv": iv - fit_iv,
        }))

    _replace_rows(con, "svi_params", snapshot_ts, pd.DataFrame(param_rows))
    resid = pd.concat(resid_frames, ignore_index=True) if resid_frames else pd.DataFrame()
    _replace_rows(con, "svi_residuals", snapshot_ts, resid)
    n_ok = sum(1 for r in param_rows if r["fit_status"] == "ok")
    log.info("SVI fits for %s: %d ok of %d expiries", snapshot_ts, n_ok, len(param_rows))
    return n_ok, len(param_rows)


def _replace_rows(
    con: duckdb.DuckDBPyConnection, table: str, snapshot_ts: datetime,
    df: pd.DataFrame,
) -> None:
    """Delete this snapshot's rows in a derived table and insert fresh ones."""
    con.execute(f"DELETE FROM {table} WHERE snapshot_ts = ?", [snapshot_ts]).fetchall()
    if df.empty:
        return
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("Float64")  # NaN -> NULL
    con.register("incoming_rows", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM incoming_rows").fetchall()
    finally:
        con.unregister("incoming_rows")


def plot_smiles(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime, underlying: str,
    n_expiries: int = 4, path=None,
):
    """Plot market IV points vs the fitted smile for a few expiries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = con.execute(
        "SELECT * FROM svi_params WHERE snapshot_ts = ? AND underlying = ? "
        "AND fit_status = 'ok' ORDER BY expiry", [snapshot_ts, underlying]).df()
    if params.empty:
        log.warning("%s: no successful fits to plot", underlying)
        return None
    picks = np.unique(np.linspace(0, len(params) - 1, min(n_expiries, len(params))).round().astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 4), squeeze=False)
    for ax, idx in zip(axes[0], picks):
        p = params.iloc[idx]
        pts = con.execute(
            "SELECT * FROM svi_residuals WHERE snapshot_ts = ? AND underlying = ? "
            "AND expiry = ?", [snapshot_ts, underlying, p["expiry"]]).df()
        t = float(pts["tte_years"].iloc[0])
        grid = np.linspace(pts["log_moneyness"].min() - 0.02, pts["log_moneyness"].max() + 0.02, 200)
        curve_iv = np.sqrt(np.maximum(svi_total_variance(
            grid, p["a"], p["b"], p["rho"], p["m"], p["sigma"]), 0) / t)
        ax.scatter(pts["log_moneyness"], pts["own_iv"], s=10, alpha=0.6, label="market (own IV)")
        ax.plot(grid, curve_iv, color="crimson", label="SVI fit")
        ax.set_title(f"{underlying} {pd.Timestamp(p['expiry']).date()}  "
                     f"({int(p['n_points'])} pts, rmse {p['rmse'] * 100:.2f} vol pts)", fontsize=9)
        ax.set_xlabel("log-moneyness k = ln(K/F)")
        ax.set_ylabel("implied vol")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    if path is None:
        config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PLOT_DIR / f"svi_{underlying}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM iv_quotes").fetchall()[0][0]
    build_svi(con, ts)
    pd.set_option("display.width", 200)
    print(f"\nSnapshot: {ts}\n\nFit status:")
    print(con.execute(
        "SELECT underlying, fit_status, count(*) AS n_expiries FROM svi_params "
        "WHERE snapshot_ts = ? GROUP BY 1, 2 ORDER BY 1, 2", [ts]).df().to_string(index=False))
    print("\nFit quality for successful fits (RMSE in vol points):")
    print(con.execute(
        "SELECT underlying, count(*) AS n_fits, sum(n_points) AS n_points, "
        "round(100*median(rmse), 2) AS median_rmse, round(100*max(rmse), 2) AS max_rmse "
        "FROM svi_params WHERE snapshot_ts = ? AND fit_status = 'ok' "
        "GROUP BY 1 ORDER BY 1", [ts]).df().to_string(index=False))
    for u in config.TICKERS:
        path = plot_smiles(con, ts, u)
        if path:
            print("saved", path)


if __name__ == "__main__":
    main()
