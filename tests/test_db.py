import duckdb
import pytest

from ivmon.db import get_connection, init_schema


def test_schema_creates_raw_quotes() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    cols = [r[0] for r in con.execute("DESCRIBE raw_quotes").fetchall()]
    assert cols == [
        "snapshot_ts", "underlying", "spot", "expiry", "option_type", "strike",
        "bid", "ask", "last", "volume", "open_interest", "vendor_iv",
        "last_trade_ts",
    ]


def test_init_schema_is_repeatable() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    init_schema(con)  # must not raise


def test_primary_key_blocks_duplicates() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    sql = ("INSERT INTO raw_quotes (snapshot_ts, underlying, expiry, "
           "option_type, strike) VALUES ('2026-01-02 10:00:00', 'SPY', "
           "'2026-02-20', 'C', 500)")
    con.execute(sql)
    with pytest.raises(duckdb.ConstraintException):
        con.execute(sql)


def test_option_type_check_constraint() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO raw_quotes (snapshot_ts, underlying, expiry, "
                    "option_type, strike) VALUES ('2026-01-02 10:00:00', "
                    "'SPY', '2026-02-20', 'X', 500)")
