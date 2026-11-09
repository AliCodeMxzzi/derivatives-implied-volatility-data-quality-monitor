"""Unsupervised ML anomaly layer: Isolation Forest on per-quote features.

Isolation Forest isolates points with random splits. Anomalies are rare and
different, so they need fewer splits to isolate (short path = high score).
No labels are needed, which suits market data where truth is unknown.
It is tree-based, so features need no scaling.

One pooled model across underlyings (more rows, one decision rule); the
residual is made relative (divided by fitted IV) so a 70%-vol stock and a
15%-vol ETF are on a comparable scale.
"""
import logging
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ivmon import config
from ivmon.db import get_connection, init_schema

log = logging.getLogger(__name__)

ML_RULE = "ml_isolation_forest"
KEY = ["underlying", "expiry", "option_type", "strike"]
FEATURES = ["rel_resid", "rel_spread", "log_moneyness", "log_tte",
            "log_vol_oi", "vendor_gap"]


def build_features(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> pd.DataFrame:
    """One row per fitted quote: key columns + FEATURES."""
    raw = con.execute(
        """
        SELECT r.underlying, r.expiry, r.option_type, r.strike,
               r.log_moneyness, r.tte_years, r.resid_iv, r.fit_iv, r.own_iv,
               i.rel_spread, i.vendor_iv, c.volume, c.open_interest
        FROM svi_residuals r
        JOIN iv_quotes i USING (snapshot_ts, underlying, expiry, option_type, strike)
        JOIN clean_quotes c USING (snapshot_ts, underlying, expiry, option_type, strike)
        WHERE r.snapshot_ts = ?
        """, [snapshot_ts]).df()
    if raw.empty:
        return pd.DataFrame(columns=KEY + FEATURES)

    out = raw[KEY].copy()
    out["rel_resid"] = raw["resid_iv"] / raw["fit_iv"]  # miss as a share of the fitted vol
    out["rel_spread"] = raw["rel_spread"].astype(float).clip(0, 5)
    out["log_moneyness"] = raw["log_moneyness"]
    out["log_tte"] = np.log(raw["tte_years"])
    volume = raw["volume"].astype(float).fillna(0.0)
    oi = raw["open_interest"].astype(float).fillna(0.0)
    out["log_vol_oi"] = np.log((volume + 1.0) / (oi + 1.0))  # volume/OI ratio, log scale
    vendor_ok = raw["vendor_iv"].astype(float) > 0.01  # tiny vendor IVs are junk
    out["vendor_gap"] = (raw["own_iv"] - raw["vendor_iv"]).where(vendor_ok)
    for col in ["rel_spread", "vendor_gap"]:  # median-fill the few missing values
        out[col] = out[col].fillna(out[col].median() if out[col].notna().any() else 0.0)
    return out.reset_index(drop=True)


def score_quotes(
    features: pd.DataFrame, contamination: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fit on the given quotes and score them. Adds ml_score and ml_flag.

    ml_score: higher = more anomalous. ml_flag: the top `contamination`
    share by score (that share is a setting, not a discovery).
    """
    contamination = config.ML_CONTAMINATION if contamination is None else contamination
    seed = config.ML_SEED if seed is None else seed
    out = features.copy()
    if len(out) < config.ML_MIN_ROWS:
        log.warning("ML skipped: only %d quotes (need %d)", len(out), config.ML_MIN_ROWS)
        out["ml_score"] = np.nan
        out["ml_flag"] = False
        return out
    model = IsolationForest(
        n_estimators=config.ML_N_ESTIMATORS, contamination=contamination,
        random_state=seed)
    labels = model.fit_predict(out[FEATURES])
    out["ml_score"] = -model.score_samples(out[FEATURES])
    out["ml_flag"] = labels == -1
    return out


def _severity(score: float, high: float, medium: float) -> str:
    if score >= high:
        return "high"
    return "medium" if score >= medium else "low"


def run_ml(
    con: duckdb.DuckDBPyConnection, snapshot_ts: datetime | None = None,
) -> pd.DataFrame:
    """Score a snapshot and store flagged quotes in `anomalies`. Returns them."""
    if snapshot_ts is None:
        snapshot_ts = con.execute("SELECT max(snapshot_ts) FROM svi_residuals").fetchall()[0][0]
    if snapshot_ts is None:
        return pd.DataFrame()
    scored = score_quotes(build_features(con, snapshot_ts))
    flagged = scored[scored["ml_flag"]].copy()

    con.execute("DELETE FROM anomalies WHERE snapshot_ts = ? AND rule = ? "
                "AND NOT is_synthetic", [snapshot_ts, ML_RULE]).fetchall()
    if not flagged.empty:
        high = scored["ml_score"].quantile(config.ML_SEVERITY_HIGH_Q)
        medium = scored["ml_score"].quantile(config.ML_SEVERITY_MEDIUM_Q)
        flagged["severity"] = [_severity(s, high, medium) for s in flagged["ml_score"]]
        flagged["details"] = [
            f"isolation score {r.ml_score:.3f} (rel_resid {r.rel_resid:+.2f}, "
            f"rel_spread {r.rel_spread:.2f}, k {r.log_moneyness:+.3f})"
            for r in flagged.itertuples()]
        flagged["snapshot_ts"] = snapshot_ts
        flagged["rule"] = ML_RULE
        flagged["is_synthetic"] = False
        flagged["strike"] = flagged["strike"].astype("Float64")
        cols = ["snapshot_ts", "underlying", "expiry", "strike", "option_type",
                "rule", "severity", "details", "is_synthetic"]
        con.register("ml_findings", flagged[cols])
        try:
            con.execute(f"INSERT INTO anomalies ({', '.join(cols)}) "
                        f"SELECT {', '.join(cols)} FROM ml_findings").fetchall()
        finally:
            con.unregister("ml_findings")
    log.info("ML flagged %d of %d quotes", len(flagged), len(scored))
    return flagged


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM svi_residuals").fetchall()[0][0]
    flagged = run_ml(con, ts)
    n = con.execute("SELECT count(*) FROM svi_residuals WHERE snapshot_ts = ?",
                    [ts]).fetchall()[0][0]
    print(f"\nSnapshot {ts}: ML flagged {len(flagged)} of {n} quotes "
          f"(contamination setting {config.ML_CONTAMINATION:.0%}: the flag rate is set, not discovered)")
    print(flagged.groupby("severity").size().to_string() if not flagged.empty else "none")


if __name__ == "__main__":
    main()
