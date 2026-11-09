from ivmon import config
from ivmon.db import get_connection, init_schema


def test_trivial() -> None:
    assert 1 + 1 == 2


def test_config_tickers() -> None:
    assert config.TICKERS == config.EQUITY_TICKERS + config.COMMODITY_PROXY_TICKERS
    assert "SPY" in config.TICKERS


def test_duckdb_in_memory() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    assert con.execute("SELECT 42").fetchone() == (42,)
