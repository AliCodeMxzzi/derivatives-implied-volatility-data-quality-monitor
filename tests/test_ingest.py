from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ivmon import config, ingest
from ivmon.db import get_connection, init_schema

TS = datetime(2026, 1, 2, 15, 0, 0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests fast: no pauses or retry waits."""
    monkeypatch.setattr(config, "REQUEST_PAUSE_SECONDS", 0)
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0)


@pytest.fixture
def con():
    c = get_connection(":memory:")
    init_schema(c)
    return c


def vendor_chain(strikes: list[float]) -> pd.DataFrame:
    """Small chain in yfinance's column format."""
    return pd.DataFrame({
        "contractSymbol": [f"X{s}" for s in strikes],
        "lastTradeDate": pd.to_datetime(
            ["2026-01-02 14:30:00+00:00"] * len(strikes)),
        "strike": strikes,
        "lastPrice": [1.0] * len(strikes),
        "bid": [0.9] * len(strikes),
        "ask": [1.1] * len(strikes),
        "volume": [10.0] * len(strikes),
        "openInterest": [100] * len(strikes),
        "impliedVolatility": [0.25] * len(strikes),
    })


class FakeSource:
    """Stands in for yfinance; no network."""

    def __init__(self, chains=None, spot=100.0, expiries=("2026-02-20",)):
        self.chains = chains or {}
        self.spot = spot
        self.expiries = list(expiries)

    def get_spot(self, ticker):
        return self.spot

    def get_expiries(self, ticker):
        return self.expiries

    def get_chain(self, ticker, expiry):
        if ticker == "BAD":
            raise RuntimeError("rate limited")
        return self.chains.get(ticker, (vendor_chain([95, 100]), vendor_chain([95, 100])))


def count(con) -> int:
    return con.execute("SELECT count(*) FROM raw_quotes").fetchone()[0]


def test_normalize_maps_columns_and_types() -> None:
    df = ingest.normalize_chain(vendor_chain([100]), TS, "SPY", 101.5, "2026-02-20", "C")
    assert list(df.columns) == ingest.RAW_COLUMNS
    row = df.iloc[0]
    assert row["vendor_iv"] == 0.25
    assert row["open_interest"] == 100
    assert row["spot"] == 101.5
    assert row["option_type"] == "C"


def test_normalize_empty_chain() -> None:
    for empty in (None, pd.DataFrame()):
        df = ingest.normalize_chain(empty, TS, "SPY", 100.0, "2026-02-20", "C")
        assert df.empty
        assert list(df.columns) == ingest.RAW_COLUMNS


def test_normalize_handles_missing_columns_and_nans() -> None:
    raw = pd.DataFrame({"strike": [100.0, np.nan, 105.0], "bid": [1.0, 2.0, np.nan]})
    df = ingest.normalize_chain(raw, TS, "SPY", None, "2026-02-20", "P")
    assert list(df["strike"]) == [100.0, 105.0]  # row with no strike dropped
    assert df["ask"].isna().all()  # missing column -> NULL
    assert df["spot"].isna().all()  # missing spot -> NULL


def test_insert_is_idempotent(con) -> None:
    df = ingest.normalize_chain(vendor_chain([95, 100]), TS, "SPY", 100.0, "2026-02-20", "C")
    assert ingest.insert_quotes(con, df) == 2
    assert ingest.insert_quotes(con, df) == 0  # exact re-run adds nothing
    assert count(con) == 2


def test_raw_values_stored_unchanged(con) -> None:
    raw = vendor_chain([100])
    raw["bid"] = 0.0  # a zero bid is bad data, but raw must keep it
    raw["ask"] = 0.5
    df = ingest.normalize_chain(raw, TS, "SPY", 100.0, "2026-02-20", "C")
    ingest.insert_quotes(con, df)
    bid, ask = con.execute("SELECT bid, ask FROM raw_quotes").fetchone()
    assert (bid, ask) == (0.0, 0.5)


def test_collect_snapshot_rerun_same_ts_is_idempotent(con) -> None:
    src = FakeSource()
    first = ingest.collect_snapshot(con, src, ["SPY"], snapshot_ts=TS)
    second = ingest.collect_snapshot(con, src, ["SPY"], snapshot_ts=TS)
    assert first == {"SPY": 4}  # 2 strikes x calls and puts
    assert second == {"SPY": 0}
    assert count(con) == 4


def test_new_snapshot_ts_adds_new_rows(con) -> None:
    src = FakeSource()
    ingest.collect_snapshot(con, src, ["SPY"], snapshot_ts=TS)
    ingest.collect_snapshot(con, src, ["SPY"], snapshot_ts=datetime(2026, 1, 3, 15, 0, 0))
    assert count(con) == 8


def test_failing_ticker_does_not_stop_others(con) -> None:
    result = ingest.collect_snapshot(con, FakeSource(), ["BAD", "SPY"], snapshot_ts=TS)
    assert result["BAD"] == 0
    assert result["SPY"] == 4


def test_empty_chain_and_no_expiries_do_not_crash(con) -> None:
    empty = {"SPY": (pd.DataFrame(), pd.DataFrame())}
    assert ingest.collect_snapshot(con, FakeSource(chains=empty), ["SPY"], snapshot_ts=TS) == {"SPY": 0}
    assert ingest.collect_snapshot(con, FakeSource(expiries=()), ["SPY"], snapshot_ts=TS) == {"SPY": 0}
    assert count(con) == 0


def test_max_expiries_limits_pull(con) -> None:
    src = FakeSource(expiries=("2026-02-20", "2026-03-20", "2026-04-17"))
    ingest.collect_snapshot(con, src, ["SPY"], snapshot_ts=TS, max_expiries=2)
    n = con.execute("SELECT count(DISTINCT expiry) FROM raw_quotes").fetchone()[0]
    assert n == 2


def test_missing_spot_stored_as_null(con) -> None:
    ingest.collect_snapshot(con, FakeSource(spot=None), ["SPY"], snapshot_ts=TS)
    assert con.execute("SELECT count(*) FROM raw_quotes WHERE spot IS NULL").fetchone()[0] == 4
