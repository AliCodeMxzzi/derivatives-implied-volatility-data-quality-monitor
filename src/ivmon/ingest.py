"""Snapshot collector: pull option chains from yfinance into raw_quotes.

Raw means raw: we only rename columns, convert types, and turn NaN into
NULL. We never filter out bad quotes here; that happens in Stage 2.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Protocol

import duckdb
import pandas as pd

from ivmon import config
from ivmon.db import get_connection, init_schema

log = logging.getLogger(__name__)

RAW_COLUMNS = [
    "snapshot_ts", "underlying", "spot", "expiry", "option_type", "strike",
    "bid", "ask", "last", "volume", "open_interest", "vendor_iv",
    "last_trade_ts",
]

# yfinance column name -> our column name
VENDOR_RENAMES = {
    "lastPrice": "last",
    "openInterest": "open_interest",
    "impliedVolatility": "vendor_iv",
    "lastTradeDate": "last_trade_ts",
}


class OptionSource(Protocol):
    """What the collector needs from a data vendor (easy to fake in tests)."""

    def get_spot(self, ticker: str) -> float | None: ...
    def get_expiries(self, ticker: str) -> list[str]: ...
    def get_chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (calls, puts) DataFrames in yfinance format."""
        ...


class YahooSource:
    """Real vendor: yfinance. Only used outside of tests."""

    def get_spot(self, ticker: str) -> float | None:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])

    def get_expiries(self, ticker: str) -> list[str]:
        import yfinance as yf

        return list(yf.Ticker(ticker).options)

    def get_chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts


def with_retries(func: Callable, *args):
    """Call a vendor function, retrying with growing waits on any error."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return func(*args)
        except Exception as exc:  # vendor errors are varied; we log and retry
            log.warning("%s%s failed (attempt %d/%d): %s",
                        func.__name__, args, attempt, config.MAX_RETRIES, exc)
            if attempt == config.MAX_RETRIES:
                raise
            time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)


def _to_utc_naive(series: pd.Series) -> pd.Series:
    """Parse timestamps to UTC and drop tz info (DuckDB TIMESTAMP is naive)."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def normalize_chain(
    raw: pd.DataFrame | None,
    snapshot_ts: datetime,
    underlying: str,
    spot: float | None,
    expiry: str,
    option_type: str,
) -> pd.DataFrame:
    """Turn one vendor chain into rows matching raw_quotes.

    Missing columns become NULL. Rows without a usable strike are dropped
    because the strike is part of the primary key.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = raw.rename(columns=VENDOR_RENAMES).copy()
    for col in ["strike", "bid", "ask", "last", "volume", "open_interest",
                "vendor_iv", "last_trade_ts"]:
        if col not in df.columns:
            df[col] = pd.NA

    out = pd.DataFrame()
    out["snapshot_ts"] = pd.Series([snapshot_ts] * len(df), dtype="datetime64[us]")
    out["underlying"] = underlying
    out["spot"] = pd.Series([spot] * len(df), dtype="Float64")
    out["expiry"] = pd.to_datetime(expiry).date()
    out["option_type"] = option_type
    for col in ["strike", "bid", "ask", "last", "vendor_iv"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    for col in ["volume", "open_interest"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    out["last_trade_ts"] = _to_utc_naive(df["last_trade_ts"])

    dropped = out["strike"].isna().sum()
    if dropped:
        log.warning("%s %s %s: dropped %d rows with no strike",
                    underlying, expiry, option_type, dropped)
    return out.dropna(subset=["strike"])[RAW_COLUMNS].reset_index(drop=True)


def _row_count(con: duckdb.DuckDBPyConnection) -> int:
    """Count raw_quotes rows (fetchall fully reads the result)."""
    return con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0]


def insert_quotes(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert rows, skipping any whose primary key already exists.

    Returns the number of rows actually added (0 on an exact re-run).
    """
    if df.empty:
        return 0
    before = _row_count(con)
    con.register("incoming", df)
    try:
        con.execute(
            f"INSERT OR IGNORE INTO raw_quotes SELECT {', '.join(RAW_COLUMNS)} "
            "FROM incoming"
        ).fetchall()  # INSERT returns a count row; read it to free the connection
    finally:
        con.unregister("incoming")
    return _row_count(con) - before


def collect_ticker(
    con: duckdb.DuckDBPyConnection,
    source: OptionSource,
    ticker: str,
    snapshot_ts: datetime,
    max_expiries: int | None,
) -> int:
    """Collect all chosen expiries for one ticker. Returns rows inserted."""
    spot = with_retries(source.get_spot, ticker)
    if spot is None:
        log.warning("%s: no spot price; storing quotes with NULL spot", ticker)

    expiries = with_retries(source.get_expiries, ticker)
    if not expiries:
        log.warning("%s: no listed expiries", ticker)
        return 0
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    total = 0
    for expiry in expiries:
        try:
            calls, puts = with_retries(source.get_chain, ticker, expiry)
        except Exception:
            log.error("%s %s: chain failed, skipping expiry", ticker, expiry)
            continue
        for raw, kind in ((calls, "C"), (puts, "P")):
            rows = normalize_chain(raw, snapshot_ts, ticker, spot, expiry, kind)
            if rows.empty:
                log.warning("%s %s %s: empty chain", ticker, expiry, kind)
            total += insert_quotes(con, rows)
        time.sleep(config.REQUEST_PAUSE_SECONDS)
    return total


def collect_snapshot(
    con: duckdb.DuckDBPyConnection,
    source: OptionSource,
    tickers: list[str],
    snapshot_ts: datetime | None = None,
    max_expiries: int | None = config.MAX_EXPIRIES,
) -> dict[str, int]:
    """Collect every ticker; one ticker failing never stops the others.

    Pass the same snapshot_ts to re-run a snapshot idempotently.
    """
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    results: dict[str, int] = {}
    for ticker in tickers:
        try:
            results[ticker] = collect_ticker(con, source, ticker, snapshot_ts, max_expiries)
        except Exception as exc:
            log.error("%s: ingestion failed: %s", ticker, exc)
            results[ticker] = 0
        log.info("%s: %d rows inserted", ticker, results[ticker])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    results = collect_snapshot(con, YahooSource(), config.TICKERS)
    print("Rows inserted per ticker:", results)


if __name__ == "__main__":
    main()
