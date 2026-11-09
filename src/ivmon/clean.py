"""Cleaning layer: derive mid/spread/time-to-expiry and flag bad quotes.

raw_quotes is never touched. clean_quotes is rebuilt from it in full every
time, so the result depends only on raw data + config thresholds.
Rows are flagged, never deleted.
"""
import logging

import duckdb
import pandas as pd

from ivmon import config
from ivmon.db import get_connection, init_schema

log = logging.getLogger(__name__)

FLAG_COLUMNS = [
    "crossed_market",
    "zero_bid",
    "wide_spread",
    "stale_trade",
    "low_liquidity",
]


def build_clean_quotes(con: duckdb.DuckDBPyConnection) -> int:
    """(Re)build the clean_quotes table. Returns its row count.

    Thresholds are numbers from config, formatted into the SQL text.
    """
    close_hour = int(config.EXPIRY_CLOSE_UTC_HOUR)
    seconds_per_year = config.DAYS_PER_YEAR * 86400.0
    stale_seconds = config.STALE_TRADE_DAYS * 86400.0
    wide = float(config.WIDE_SPREAD_REL)

    sql = f"""
    CREATE OR REPLACE TABLE clean_quotes AS
    WITH base AS (
        SELECT
            snapshot_ts, underlying, spot, expiry, option_type, strike,
            bid, ask, last, volume, open_interest, vendor_iv, last_trade_ts,
            (bid + ask) / 2.0 AS mid,
            ask - bid AS spread,
            date_diff('second', snapshot_ts,
                      CAST(expiry AS TIMESTAMP) + INTERVAL {close_hour} HOUR)
                / {seconds_per_year} AS tte_years
        FROM raw_quotes
    ),
    with_rel AS (
        SELECT *, CASE WHEN mid > 0 THEN spread / mid END AS rel_spread
        FROM base
    )
    SELECT
        *,
        COALESCE(bid > ask, FALSE) AS crossed_market,
        COALESCE(bid, 0) <= 0 AS zero_bid,
        COALESCE(rel_spread > {wide}, FALSE) AS wide_spread,
        (last_trade_ts IS NULL
         OR date_diff('second', last_trade_ts, snapshot_ts) > {stale_seconds})
            AS stale_trade,
        (COALESCE(volume, 0) = 0 AND COALESCE(open_interest, 0) = 0)
            AS low_liquidity
    FROM with_rel
    """
    con.execute(sql).fetchall()
    n = con.execute("SELECT count(*) FROM clean_quotes").fetchall()[0][0]
    log.info("clean_quotes rebuilt: %d rows", n)
    return n


def flag_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Percent of quotes flagged per rule, per underlying, plus an ALL row."""
    pct_cols = ",\n        ".join(
        f"round(100.0 * avg(CAST({f} AS INTEGER)), 1) AS {f}"
        for f in FLAG_COLUMNS
    )
    sql = f"""
    SELECT
        COALESCE(underlying, 'ALL') AS underlying,
        count(*) AS n_quotes,
        {pct_cols}
    FROM clean_quotes
    GROUP BY ROLLUP (underlying)
    ORDER BY underlying = 'ALL', underlying
    """
    return con.execute(sql).df()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    init_schema(con)
    build_clean_quotes(con)
    pd.set_option("display.width", 200)
    print(flag_summary(con).to_string(index=False))


if __name__ == "__main__":
    main()
