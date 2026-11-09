import pandas as pd
import pytest

from ivmon import clean, config, iv
from ivmon.db import get_connection, init_schema
from ivmon.forward import spot_forward
from ivmon.pricing import black_price

SNAP = "2026-01-02 21:00:00"
EXPIRY = "2026-04-03"  # exactly 91 days after SNAP -> T = 91/365
T = 91 / 365.0
SPOT = 100.0
TRUE_F = 101.0
SIGMA = 0.25
STRIKES = [85, 90, 95, 100, 105, 110, 115]


@pytest.fixture
def con():
    c = get_connection(":memory:")
    init_schema(c)
    return c


def add(con, strike, kind, bid, ask, underlying="SPY", vendor_iv=0.30) -> None:
    con.execute(
        "INSERT INTO raw_quotes (snapshot_ts, underlying, spot, expiry, "
        "option_type, strike, bid, ask, last, volume, open_interest, "
        "vendor_iv, last_trade_ts) VALUES "
        f"('{SNAP}', '{underlying}', {SPOT}, '{EXPIRY}', '{kind}', {strike}, "
        f"{bid}, {ask}, 1.0, 10, 100, {vendor_iv}, '2026-01-02 20:00:00')"
    ).fetchall()


def add_model_chain(con, strikes=STRIKES, kinds=("C", "P")) -> None:
    for k in strikes:
        for kind in kinds:
            mid = black_price(TRUE_F, k, T, SIGMA, config.RISK_FREE_RATE, kind)
            add(con, k, kind, mid - 0.001, mid + 0.001)


def build(con):
    clean.build_clean_quotes(con)
    n = iv.build_iv_quotes(con)
    return n, con.execute("SELECT * FROM iv_quotes ORDER BY strike, option_type").df()


def test_only_otm_options_are_kept(con) -> None:
    add_model_chain(con)
    n, df = build(con)
    assert n == 7  # forward 101: puts 85-100 (4) + calls 105-115 (3)
    puts = df[df["option_type"] == "P"]
    calls = df[df["option_type"] == "C"]
    assert (puts["strike"] < df["forward"].iloc[0]).all()
    assert (calls["strike"] >= df["forward"].iloc[0]).all()


def test_forward_from_parity_and_iv_recovered(con) -> None:
    add_model_chain(con)
    _, df = build(con)
    assert (df["forward_method"] == "parity").all()
    assert df["forward"].iloc[0] == pytest.approx(TRUE_F, abs=1e-6)
    assert (df["iv_reason"] == "ok").all()
    assert df["own_iv"].tolist() == pytest.approx([SIGMA] * 7, abs=1e-6)


def test_iv_diff_is_own_minus_vendor(con) -> None:
    add_model_chain(con)
    _, df = build(con)
    expected = df["own_iv"] - df["vendor_iv"]
    assert df["iv_diff"].tolist() == pytest.approx(expected.tolist())


def test_zero_bid_quote_is_skipped_with_reason(con) -> None:
    add_model_chain(con)
    add(con, 120, "C", 0, 0.05)
    _, df = build(con)
    bad = df[df["strike"] == 120].iloc[0]
    assert bad["iv_reason"] == "skipped_zero_bid"
    assert pd.isna(bad["own_iv"])


def test_crossed_quote_is_skipped(con) -> None:
    add_model_chain(con)
    add(con, 120, "C", 0.50, 0.30)
    _, df = build(con)
    assert df[df["strike"] == 120].iloc[0]["iv_reason"] == "skipped_crossed_market"


def test_fallback_forward_when_no_puts(con) -> None:
    add_model_chain(con, kinds=("C",))
    _, df = build(con)
    assert (df["forward_method"] == "spot_fallback").all()
    assert df["forward"].iloc[0] == pytest.approx(spot_forward(SPOT, T))


def test_rebuild_is_idempotent(con) -> None:
    add_model_chain(con)
    clean.build_clean_quotes(con)
    first = iv.build_iv_quotes(con)
    second = iv.build_iv_quotes(con)
    total = con.execute("SELECT count(*) FROM iv_quotes").fetchall()[0][0]
    assert first == second == total == 7


def test_reason_counts_and_comparison(con) -> None:
    add_model_chain(con)
    add(con, 120, "C", 0, 0.05)
    clean.build_clean_quotes(con)
    iv.build_iv_quotes(con)
    ts = con.execute("SELECT max(snapshot_ts) FROM iv_quotes").fetchall()[0][0]
    reasons = iv.reason_counts(con, ts).set_index("iv_reason")["n"]
    assert reasons["ok"] == 7
    assert reasons["skipped_zero_bid"] == 1
    comp = iv.iv_comparison(con, ts).iloc[0]
    assert comp["n_otm"] == 8
    assert comp["n_solved"] == 7
    assert comp["n_compared"] == 7
    # own 25% vs vendor 30% -> median diff -5 vol points
    assert comp["median_diff"] == pytest.approx(-5.0, abs=0.01)
