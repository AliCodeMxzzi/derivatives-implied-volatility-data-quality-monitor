"""DuckDB connection and schema management.

No tables yet: each stage adds the tables it needs to `init_schema`.
"""
from pathlib import Path

import duckdb

from ivmon import config


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection. Use ':memory:' for tests."""
    path = str(db_path if db_path is not None else config.DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they do not exist. Empty for Stage 0."""
    # Intentionally empty. Stage 1 adds raw_quotes.
