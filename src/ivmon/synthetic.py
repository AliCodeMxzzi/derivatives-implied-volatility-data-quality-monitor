"""Synthetic anomaly injection and honest evaluation.

We copy ONE real snapshot into a separate DuckDB, corrupt chosen quotes in a
known way, run the whole pipeline on the copy, and ask which detectors found
the planted quotes. The real database is only read, never written.

Limits (state these when you present results):
- Corruption types and sizes are my choices; recall depends on them.
- Planted errors are simpler and cleaner than real vendor errors.
- Quotes that were already odd in the real data count as false positives
  here; a no-injection baseline run shows how many flags existed anyway.
"""
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ivmon import config
from ivmon.checks import run_checks
from ivmon.clean import build_clean_quotes
from ivmon.db import get_connection, init_schema
from ivmon.ingest import RAW_COLUMNS, insert_quotes
from ivmon.iv import build_iv_quotes
from ivmon.ml import ML_RULE, run_ml
from ivmon.pricing import black_price
from ivmon.svi import build_svi

log = logging.getLogger(__name__)

INJECTION_TYPES = ["price_spike", "stale_quote", "swapped_bid_ask", "strike_typo", "iv_shift"]
KEY = ["underlying", "expiry", "option_type", "strike"]

INJECTIONS_DDL = """
CREATE TABLE IF NOT EXISTS injections (
    snapshot_ts     TIMESTAMP NOT NULL,
    underlying      VARCHAR   NOT NULL,
    expiry          DATE      NOT NULL,
    option_type     VARCHAR   NOT NULL,
    strike          DOUBLE    NOT NULL,   -- where the corrupted quote now lives
    original_strike DOUBLE    NOT NULL,
    injection_type  VARCHAR   NOT NULL,
    original_bid    DOUBLE,
    original_ask    DOUBLE,
    new_bid         DOUBLE,
    new_ask         DOUBLE,
    details         VARCHAR,
    is_synthetic    BOOLEAN   NOT NULL DEFAULT TRUE
);
"""


# ------------------------------------------------------------------ injection

def select_targets(
    real_con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
    n_per_type: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick distinct healthy quotes (solved IV, no quote flags), n per type."""
    healthy = real_con.execute(
        """
        SELECT underlying, expiry, option_type, strike, forward, tte_years,
               own_iv, rel_spread
        FROM iv_quotes
        WHERE snapshot_ts = ? AND own_iv IS NOT NULL AND rel_spread > 0
          AND NOT crossed_market AND NOT zero_bid AND NOT wide_spread
        """, [snapshot_ts]).df()
    n_per_type = min(n_per_type, len(healthy) // len(INJECTION_TYPES))
    picked = healthy.iloc[rng.permutation(len(healthy))[: n_per_type * len(INJECTION_TYPES)]].copy()
    picked["injection_type"] = np.repeat(INJECTION_TYPES, n_per_type)
    return picked.reset_index(drop=True)


def transposed_strike(strike: float, taken: set, prefix: tuple) -> float | None:
    """A 'typo' strike: swap two adjacent digits of the integer part.

    Tries the first two digits (big move), then the last two (small move).
    Returns None if no valid, unused strike results.
    """
    digits = str(int(round(strike)))
    if len(digits) < 2:
        return None
    for i in (0, len(digits) - 2):
        swapped = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
        candidate = float(int(swapped))
        if candidate >= 1 and candidate != strike and (*prefix, candidate) not in taken:
            return candidate
    return None


def inject(
    raw: pd.DataFrame, targets: pd.DataFrame, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply corruptions to a COPY of raw quotes. Returns (corrupted, truth)."""
    raw = raw.copy()
    taken = set(zip(raw["underlying"], raw["expiry"], raw["option_type"], raw["strike"]))
    truth = []
    for t in targets.itertuples():
        mask = ((raw["underlying"] == t.underlying) & (raw["expiry"] == t.expiry)
                & (raw["option_type"] == t.option_type) & (raw["strike"] == t.strike))
        if mask.sum() != 1:
            continue
        i = raw.index[mask][0]
        bid0, ask0 = float(raw.at[i, "bid"]), float(raw.at[i, "ask"])
        bid, ask, note = bid0, ask0, ""
        new_strike = float(t.strike)

        if t.injection_type == "price_spike":
            f = float(rng.choice(config.SYNTH_SPIKE_FACTORS))
            bid, ask, note = bid0 * f, ask0 * f, f"price x{f}"
        elif t.injection_type == "stale_quote":
            s = float(rng.choice(config.SYNTH_STALE_SHIFTS))
            bid, ask = bid0 * s, ask0 * s
            old_ts = pd.Timestamp(raw.at[i, "last_trade_ts"])
            if pd.notna(old_ts):
                raw.at[i, "last_trade_ts"] = old_ts - pd.Timedelta(days=config.SYNTH_STALE_DAYS)
            note = f"price x{s}, last trade {config.SYNTH_STALE_DAYS}d older"
        elif t.injection_type == "swapped_bid_ask":
            bid, ask, note = ask0, bid0, "bid and ask swapped"
        elif t.injection_type == "strike_typo":
            candidate = transposed_strike(t.strike, taken, (t.underlying, t.expiry, t.option_type))
            if candidate is None:
                continue
            taken.discard((t.underlying, t.expiry, t.option_type, t.strike))
            taken.add((t.underlying, t.expiry, t.option_type, candidate))
            new_strike = candidate
            raw.at[i, "strike"] = candidate
            note = f"strike {t.strike:g} -> {candidate:g}"
        elif t.injection_type == "iv_shift":
            f = float(rng.choice(config.SYNTH_IV_FACTORS))
            mid = black_price(t.forward, t.strike, t.tte_years, t.own_iv * f,
                              config.RISK_FREE_RATE, t.option_type)
            half = t.rel_spread / 2.0  # keep the relative spread unchanged
            bid, ask, note = mid * (1 - half), mid * (1 + half), f"IV x{f}"

        raw.at[i, "bid"], raw.at[i, "ask"] = bid, ask
        truth.append({
            "snapshot_ts": raw.at[i, "snapshot_ts"], "underlying": t.underlying,
            "expiry": t.expiry, "option_type": t.option_type, "strike": new_strike,
            "original_strike": float(t.strike), "injection_type": t.injection_type,
            "original_bid": bid0, "original_ask": ask0, "new_bid": bid, "new_ask": ask,
            "details": note, "is_synthetic": True,
        })
    return raw, pd.DataFrame(truth)


# ------------------------------------------------------------------- pipeline

def run_pipeline(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> None:
    """Everything after ingestion, on one snapshot."""
    build_clean_quotes(con)
    build_iv_quotes(con, snapshot_ts)
    build_svi(con, snapshot_ts)
    run_checks(con, snapshot_ts)
    run_ml(con, snapshot_ts)


def _load_raw(real_con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> pd.DataFrame:
    return real_con.execute(
        "SELECT * FROM raw_quotes WHERE snapshot_ts = ?", [snapshot_ts]).df()


def _fresh_db(path: Path | str) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:":
        for suffix in ("", ".wal"):
            Path(str(path) + suffix).unlink(missing_ok=True)
    con = get_connection(path)
    init_schema(con)
    con.execute(INJECTIONS_DDL)
    return con


def build_synthetic(
    real_con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
    db_path: Path | str | None = None, n_per_type: int | None = None,
    seed: int | None = None,
) -> tuple[duckdb.DuckDBPyConnection, pd.DataFrame]:
    """Create the corrupted copy, run the pipeline on it, mark findings synthetic."""
    seed = config.SYNTH_SEED if seed is None else seed
    n_per_type = config.SYNTH_N_PER_TYPE if n_per_type is None else n_per_type
    rng = np.random.default_rng(seed)
    raw = _load_raw(real_con, snapshot_ts)
    targets = select_targets(real_con, snapshot_ts, n_per_type, rng)
    corrupted, truth = inject(raw, targets, rng)

    con = _fresh_db(db_path if db_path is not None else config.SYNTH_DB_PATH)
    insert_quotes(con, corrupted[RAW_COLUMNS])
    if not truth.empty:
        con.register("truth_in", truth)
        con.execute("INSERT INTO injections SELECT snapshot_ts, underlying, expiry, "
                    "option_type, strike, original_strike, injection_type, original_bid, "
                    "original_ask, new_bid, new_ask, details, is_synthetic FROM truth_in").fetchall()
        con.unregister("truth_in")
    run_pipeline(con, snapshot_ts)
    con.execute("UPDATE anomalies SET is_synthetic = TRUE").fetchall()
    return con, truth


def build_baseline(
    real_con: duckdb.DuckDBPyConnection, snapshot_ts: datetime,
) -> duckdb.DuckDBPyConnection:
    """Same pipeline on the UNcorrupted copy (in memory): how much gets flagged anyway."""
    con = _fresh_db(":memory:")
    insert_quotes(con, _load_raw(real_con, snapshot_ts)[RAW_COLUMNS])
    run_pipeline(con, snapshot_ts)
    return con


# ----------------------------------------------------------------- evaluation

def _key(underlying, expiry, option_type, strike) -> tuple:
    return (underlying, pd.Timestamp(expiry).date(), option_type, float(strike))


def flagged_sets(con: duckdb.DuckDBPyConnection, snapshot_ts: datetime) -> dict[str, set]:
    """Quote keys flagged by the rule-based layer, the ML layer, and both.

    Rule-based = Stage 5 findings + the Stage 2 crossed_market flag.
    Pair-level parity findings (no option_type) flag both legs at that strike.
    Expiry-level findings (no strike) cannot point at a quote, so they add none.
    """
    rules: set = set()
    ml: set = set()
    rows = con.execute(
        "SELECT underlying, expiry, strike, option_type, rule FROM anomalies "
        "WHERE snapshot_ts = ?", [snapshot_ts]).fetchall()
    for underlying, expiry, strike, option_type, rule in rows:
        if strike is None:
            continue
        kinds = [option_type] if option_type else ["C", "P"]
        target = ml if rule == ML_RULE else rules
        for kind in kinds:
            target.add(_key(underlying, expiry, kind, strike))
    for underlying, expiry, option_type, strike in con.execute(
            "SELECT underlying, expiry, option_type, strike FROM clean_quotes "
            "WHERE snapshot_ts = ? AND crossed_market", [snapshot_ts]).fetchall():
        rules.add(_key(underlying, expiry, option_type, strike))
    return {"rule_based": rules, "ml": ml, "combined": rules | ml}


def score_detector(flagged: set, truth: set, universe: int) -> dict:
    """Quote-level confusion counts, precision and recall."""
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    tn = universe - tp - fp - fn
    return {
        "flagged": len(flagged), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def evaluate(
    syn_con: duckdb.DuckDBPyConnection, base_con: duckdb.DuckDBPyConnection,
    snapshot_ts: datetime,
) -> dict[str, pd.DataFrame]:
    """Metrics per detector, recall per corruption type, and the baseline flag counts."""
    truth_df = syn_con.execute("SELECT * FROM injections").df()
    truth = {_key(r.underlying, r.expiry, r.option_type, r.strike) for r in truth_df.itertuples()}
    universe = syn_con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0]
    syn = flagged_sets(syn_con, snapshot_ts)
    base = flagged_sets(base_con, snapshot_ts)

    rows = []
    for name, flagged in syn.items():
        rows.append({"detector": name, **score_detector(flagged, truth, universe),
                     "flagged_without_injection": len(base[name])})
    summary = pd.DataFrame(rows)

    by_type = []
    for kind, g in truth_df.groupby("injection_type"):
        keys = {_key(r.underlying, r.expiry, r.option_type, r.strike) for r in g.itertuples()}
        by_type.append({"injection_type": kind, "n_injected": len(keys),
                        **{f"recall_{name}": len(keys & flagged) / len(keys)
                           for name, flagged in syn.items()}})
    return {"summary": summary, "recall_by_type": pd.DataFrame(by_type)}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    real = get_connection(config.DB_PATH)
    init_schema(real)
    ts = real.execute("SELECT max(snapshot_ts) FROM svi_params").fetchall()[0][0]
    syn_con, truth = build_synthetic(real, ts)
    base_con = build_baseline(real, ts)
    result = evaluate(syn_con, base_con, ts)
    pd.set_option("display.width", 220)
    n_raw = syn_con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0]
    print(f"\nSnapshot {ts}: {len(truth)} quotes injected into {n_raw} (seed {config.SYNTH_SEED})")
    print(f"ML contamination setting: {config.ML_CONTAMINATION:.0%} (flag rate is set, not discovered)")
    print("\nInjected, by type:")
    print(truth.groupby("injection_type").size().to_string())
    print("\nDetection summary (quote level):")
    print(result["summary"].round(3).to_string(index=False))
    print("\nRecall by corruption type:")
    print(result["recall_by_type"].round(2).to_string(index=False))
    print(f"\nSynthetic results are in {config.SYNTH_DB_PATH} (all findings is_synthetic = TRUE).")


if __name__ == "__main__":
    main()
