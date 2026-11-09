import pytest

from ivmon import clean, config
from ivmon.db import get_connection, init_schema

SNAP = "2026-01-02 21:00:00"


@pytest.fixture
def con():
    c = get_connection(":memory:")
    init_schema(c)
    return c


def add_quote(con, strike: float = 100.0, underlying: str = "SPY",
              expiry: str = "2026-01-12", option_type: str = "C",
              bid="1.00", ask="1.10", volume="10", open_interest="100",
              last_trade_ts="'2026-01-02 20:00:00'") -> None:
    """Insert one hand-built raw quote. Defaults describe a healthy quote."""
    con.execute(
        "INSERT INTO raw_quotes (snapshot_ts, underlying, spot, expiry, "
        "option_type, strike, bid, ask, last, volume, open_interest, "
        "vendor_iv, last_trade_ts) VALUES "
        f"('{SNAP}', '{underlying}', 100, '{expiry}', '{option_type}', "
        f"{strike}, {bid}, {ask}, 1.05, {volume}, {open_interest}, 0.2, "
        f"{last_trade_ts})"
    ).fetchall()


def row(con, strike: float = 100.0) -> dict:
    df = con.execute(
        f"SELECT * FROM clean_quotes WHERE strike = {strike}").df()
    return df.iloc[0].to_dict()


def test_healthy_quote_has_no_flags(con) -> None:
    add_quote(con)
    clean.build_clean_quotes(con)
    r = row(con)
    assert not any(r[f] for f in clean.FLAG_COLUMNS)


def test_mid_spread_rel_spread(con) -> None:
    add_quote(con, bid="1.00", ask="1.20")
    clean.build_clean_quotes(con)
    r = row(con)
    assert r["mid"] == pytest.approx(1.10)
    assert r["spread"] == pytest.approx(0.20)
    assert r["rel_spread"] == pytest.approx(0.20 / 1.10)


def test_time_to_expiry_act_365(con) -> None:
    # snapshot 2026-01-02 21:00 UTC, expiry 2026-01-12 at 21:00 UTC = 10 days
    add_quote(con)
    clean.build_clean_quotes(con)
    assert row(con)["tte_years"] == pytest.approx(10 / 365.0)


def test_crossed_market(con) -> None:
    add_quote(con, strike=100, bid="1.20", ask="1.00")
    add_quote(con, strike=105, bid="1.00", ask="1.00")  # locked, not crossed
    clean.build_clean_quotes(con)
    assert row(con, 100)["crossed_market"]
    assert not row(con, 105)["crossed_market"]


def test_zero_bid_and_null_bid(con) -> None:
    add_quote(con, strike=100, bid="0", ask="0.05")
    add_quote(con, strike=105, bid="NULL", ask="0.05")
    add_quote(con, strike=110, bid="0.01", ask="0.05")
    clean.build_clean_quotes(con)
    assert row(con, 100)["zero_bid"]
    assert row(con, 105)["zero_bid"]
    assert not row(con, 110)["zero_bid"]


def test_wide_spread_threshold(con) -> None:
    # rel spread = 0.2/1.0 = 0.2 (ok) vs 1.0/1.0 = 1.0 (wide)
    add_quote(con, strike=100, bid="0.90", ask="1.10")
    add_quote(con, strike=105, bid="0.50", ask="1.50")
    clean.build_clean_quotes(con)
    assert not row(con, 100)["wide_spread"]
    assert row(con, 105)["wide_spread"]


def test_wide_spread_false_when_mid_is_zero(con) -> None:
    add_quote(con, bid="0", ask="0")  # mid 0 -> relative spread undefined
    clean.build_clean_quotes(con)
    r = row(con)
    assert r["wide_spread"] in (False, 0)
    assert r["zero_bid"]


def test_stale_trade(con) -> None:
    add_quote(con, strike=100, last_trade_ts="'2026-01-01 15:00:00'")  # ~1 day
    add_quote(con, strike=105, last_trade_ts="'2025-12-20 15:00:00'")  # 13 days
    add_quote(con, strike=110, last_trade_ts="NULL")
    clean.build_clean_quotes(con)
    assert not row(con, 100)["stale_trade"]
    assert row(con, 105)["stale_trade"]
    assert row(con, 110)["stale_trade"]


def test_low_liquidity(con) -> None:
    add_quote(con, strike=100, volume="0", open_interest="0")
    add_quote(con, strike=105, volume="0", open_interest="50")
    add_quote(con, strike=110, volume="NULL", open_interest="NULL")
    clean.build_clean_quotes(con)
    assert row(con, 100)["low_liquidity"]
    assert not row(con, 105)["low_liquidity"]
    assert row(con, 110)["low_liquidity"]


def test_no_rows_deleted_and_rebuild_is_stable(con) -> None:
    add_quote(con, strike=100, bid="1.20", ask="1.00")  # crossed
    add_quote(con, strike=105, bid="0", ask="0")  # junk
    add_quote(con, strike=110)
    assert clean.build_clean_quotes(con) == 3
    assert clean.build_clean_quotes(con) == 3  # rebuild, same result
    raw_n = con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0]
    assert raw_n == 3  # raw untouched


def test_flag_summary_percentages(con) -> None:
    add_quote(con, strike=100, underlying="SPY")
    add_quote(con, strike=105, underlying="SPY", bid="1.20", ask="1.00")
    add_quote(con, strike=100, underlying="GLD", volume="0", open_interest="0")
    clean.build_clean_quotes(con)
    s = clean.flag_summary(con).set_index("underlying")
    assert s.loc["SPY", "n_quotes"] == 2
    assert s.loc["SPY", "crossed_market"] == 50.0
    assert s.loc["GLD", "low_liquidity"] == 100.0
    assert s.loc["ALL", "n_quotes"] == 3
    assert s.index[-1] == "ALL"


def test_thresholds_come_from_config(con, monkeypatch) -> None:
    add_quote(con, bid="0.90", ask="1.10")  # rel spread 0.2
    monkeypatch.setattr(config, "WIDE_SPREAD_REL", 0.1)
    clean.build_clean_quotes(con)
    assert row(con)["wide_spread"]
