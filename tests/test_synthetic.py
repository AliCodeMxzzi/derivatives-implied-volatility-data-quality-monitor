import numpy as np
import pandas as pd
import pytest

from ivmon import config, synthetic
from ivmon.clean import build_clean_quotes
from ivmon.db import get_connection, init_schema
from ivmon.ingest import RAW_COLUMNS
from ivmon.pricing import implied_vol

SNAP = pd.Timestamp("2026-01-02 21:00:00")
EXP = pd.Timestamp("2026-04-03")


def raw_frame() -> pd.DataFrame:
    rows = []
    for k in (95.0, 100.0, 105.0, 110.0):
        rows.append({
            "snapshot_ts": SNAP, "underlying": "SPY", "spot": 100.0, "expiry": EXP,
            "option_type": "C", "strike": k, "bid": 1.00, "ask": 1.20, "last": 1.1,
            "volume": 10, "open_interest": 100, "vendor_iv": 0.25,
            "last_trade_ts": pd.Timestamp("2026-01-02 20:00:00")})
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def target(kind: str, strike: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "underlying": "SPY", "expiry": EXP, "option_type": "C", "strike": strike,
        "forward": 101.0, "tte_years": 91 / 365, "own_iv": 0.25,
        "rel_spread": 0.2 / 1.1, "injection_type": kind}])


def run_inject(kind: str, strike: float = 100.0, seed: int = 0):
    raw = raw_frame()
    before = raw.copy()
    corrupted, truth = synthetic.inject(raw, target(kind, strike), np.random.default_rng(seed))
    pd.testing.assert_frame_equal(raw, before)  # input is never modified
    return corrupted, truth


def corrupted_row(corrupted: pd.DataFrame, strike: float) -> pd.Series:
    return corrupted[corrupted["strike"] == strike].iloc[0]


# --------------------------------------------------------------- injection

def test_price_spike_scales_both_sides() -> None:
    corrupted, truth = run_inject("price_spike")
    r = corrupted_row(corrupted, 100.0)
    factor = r["bid"] / 1.00
    assert factor in config.SYNTH_SPIKE_FACTORS
    assert r["ask"] == pytest.approx(1.20 * factor)
    assert truth.iloc[0]["injection_type"] == "price_spike"


def test_stale_quote_shifts_price_and_ages_trade() -> None:
    corrupted, _ = run_inject("stale_quote")
    r = corrupted_row(corrupted, 100.0)
    assert r["bid"] / 1.00 in config.SYNTH_STALE_SHIFTS
    assert r["last_trade_ts"] == pd.Timestamp("2026-01-02 20:00:00") - pd.Timedelta(
        days=config.SYNTH_STALE_DAYS)


def test_swapped_bid_ask_is_crossed() -> None:
    corrupted, _ = run_inject("swapped_bid_ask")
    r = corrupted_row(corrupted, 100.0)
    assert (r["bid"], r["ask"]) == (1.20, 1.00)
    assert r["bid"] > r["ask"]


def test_strike_typo_moves_the_quote_and_truth_uses_new_strike() -> None:
    corrupted, truth = run_inject("strike_typo", strike=100.0)
    new_strike = truth.iloc[0]["strike"]
    assert new_strike != 100.0
    assert new_strike in set(corrupted["strike"])
    assert 100.0 not in set(corrupted["strike"])
    assert truth.iloc[0]["original_strike"] == 100.0
    assert corrupted["strike"].is_unique  # no key collision created


def test_iv_shift_changes_iv_by_the_factor_and_keeps_relative_spread() -> None:
    corrupted, truth = run_inject("iv_shift")
    r = corrupted_row(corrupted, 100.0)
    mid = (r["bid"] + r["ask"]) / 2
    result = implied_vol(mid, 101.0, 100.0, 91 / 365, config.RISK_FREE_RATE, "C")
    assert result.reason == "ok"
    factor = result.iv / 0.25
    assert min(abs(factor - f) for f in config.SYNTH_IV_FACTORS) < 1e-6
    assert (r["ask"] - r["bid"]) / mid == pytest.approx(0.2 / 1.1)


def test_injection_is_reproducible_and_flagged_synthetic() -> None:
    a, truth_a = run_inject("price_spike", seed=5)
    b, truth_b = run_inject("price_spike", seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert truth_a["is_synthetic"].all()


def test_transposed_strike_rules() -> None:
    prefix = ("SPY", EXP, "C")
    assert synthetic.transposed_strike(487.0, set(), prefix) == 847.0  # big move first
    taken = {(*prefix, 847.0)}
    assert synthetic.transposed_strike(487.0, taken, prefix) == 478.0  # falls back
    assert synthetic.transposed_strike(111.0, set(), prefix) is None  # no change possible
    assert synthetic.transposed_strike(5.0, set(), prefix) is None  # single digit


# -------------------------------------------------------------- evaluation

def test_score_detector_hand_computed() -> None:
    m = synthetic.score_detector({1, 2, 3, 4}, {3, 4, 5}, universe=10)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (2, 2, 1, 5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(2 / 3)


def test_score_detector_empty_cases() -> None:
    m = synthetic.score_detector(set(), {1}, universe=5)
    assert m["recall"] == 0.0
    assert np.isnan(m["precision"])


def test_flagged_sets_semantics() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    # one crossed raw quote -> clean_quotes crossed_market flag
    con.execute("INSERT INTO raw_quotes (snapshot_ts, underlying, spot, expiry, "
                "option_type, strike, bid, ask) VALUES ('2026-01-02 21:00:00', 'SPY', "
                "100, '2026-04-03', 'C', 120, 1.5, 1.0)").fetchall()
    build_clean_quotes(con)

    def add(strike, kind, rule):
        s = "NULL" if strike is None else strike
        t = "NULL" if kind is None else f"'{kind}'"
        con.execute("INSERT INTO anomalies VALUES ('2026-01-02 21:00:00', 'SPY', "
                    f"'2026-04-03', {s}, {t}, '{rule}', 'low', 'x', FALSE)").fetchall()

    add(100, "P", "residual_outlier")
    add(105, None, "put_call_parity")  # pair-level: both legs
    add(None, None, "butterfly_arbitrage")  # expiry-level: flags no quote
    add(110, "C", synthetic.ML_RULE)
    sets = synthetic.flagged_sets(con, pd.Timestamp("2026-01-02 21:00:00"))
    d = pd.Timestamp("2026-04-03").date()
    assert sets["rule_based"] == {
        ("SPY", d, "P", 100.0), ("SPY", d, "C", 105.0), ("SPY", d, "P", 105.0),
        ("SPY", d, "C", 120.0)}  # last one from crossed_market
    assert sets["ml"] == {("SPY", d, "C", 110.0)}
    assert sets["combined"] == sets["rule_based"] | sets["ml"]


# ----------------------------------------------------------------- end to end

def test_end_to_end_isolation_and_evaluation(model_con, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_MIN_ROWS", 10)
    real, ts = model_con
    raw_before = real.execute("SELECT count(*), sum(bid) FROM raw_quotes").fetchall()
    anomalies_before = real.execute("SELECT count(*) FROM anomalies").fetchall()[0][0]

    syn, truth = synthetic.build_synthetic(real, ts, db_path=":memory:", n_per_type=2, seed=1)
    base = synthetic.build_baseline(real, ts)

    # 2 per type x 5 types; a strike typo may be skipped if no valid strike exists
    assert 8 <= len(truth) <= 10
    assert syn.execute("SELECT count(*) FROM injections").fetchall()[0][0] == len(truth)
    # everything found in the synthetic DB is marked synthetic
    assert syn.execute("SELECT count(*) FROM anomalies WHERE NOT is_synthetic").fetchall()[0][0] == 0
    # the real database was only read
    assert real.execute("SELECT count(*), sum(bid) FROM raw_quotes").fetchall() == raw_before
    assert real.execute("SELECT count(*) FROM anomalies").fetchall()[0][0] == anomalies_before

    result = synthetic.evaluate(syn, base, ts)
    assert list(result["summary"]["detector"]) == ["rule_based", "ml", "combined"]
    by_type = result["recall_by_type"].set_index("injection_type")
    # a swapped bid/ask is crossed by construction, so the rule layer must catch it
    assert by_type.loc["swapped_bid_ask", "recall_rule_based"] == 1.0
    s = result["summary"].set_index("detector")
    assert s.loc["combined", "TP"] >= max(s.loc["rule_based", "TP"], s.loc["ml", "TP"])
