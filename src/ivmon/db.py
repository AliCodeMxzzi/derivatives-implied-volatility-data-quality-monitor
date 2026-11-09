"""DuckDB connection and schema management.

Each stage adds the tables it needs to `init_schema`.
"""
from pathlib import Path

import duckdb

from ivmon import config

# Raw layer: exactly what the vendor sent. Never updated or deleted by
# cleaning code. The primary key makes re-running a snapshot idempotent.
RAW_QUOTES_DDL = """
CREATE TABLE IF NOT EXISTS raw_quotes (
    snapshot_ts    TIMESTAMP NOT NULL,   -- when WE pulled the data (UTC)
    underlying     VARCHAR   NOT NULL,
    spot           DOUBLE,               -- underlying price at snapshot time
    expiry         DATE      NOT NULL,
    option_type    VARCHAR   NOT NULL CHECK (option_type IN ('C', 'P')),
    strike         DOUBLE    NOT NULL,
    bid            DOUBLE,
    ask            DOUBLE,
    last           DOUBLE,
    volume         BIGINT,
    open_interest  BIGINT,
    vendor_iv      DOUBLE,               -- vendor's implied vol, kept for comparison
    last_trade_ts  TIMESTAMP,            -- UTC
    PRIMARY KEY (snapshot_ts, underlying, expiry, option_type, strike)
);
"""


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection. Use ':memory:' for tests."""
    path = str(db_path if db_path is not None else config.DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    con.execute(RAW_QUOTES_DDL)
