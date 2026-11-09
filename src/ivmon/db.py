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


# Derived layer: our own IV for out-of-the-money quotes, next to vendor IV.
# Rebuilt per snapshot from clean_quotes, so it can always be regenerated.
IV_QUOTES_DDL = """
CREATE TABLE IF NOT EXISTS iv_quotes (
    snapshot_ts     TIMESTAMP NOT NULL,
    underlying      VARCHAR   NOT NULL,
    expiry          DATE      NOT NULL,
    option_type     VARCHAR   NOT NULL,
    strike          DOUBLE    NOT NULL,
    spot            DOUBLE,
    forward         DOUBLE,
    forward_method  VARCHAR,             -- parity / spot_fallback / no_forward
    tte_years       DOUBLE,
    mid             DOUBLE,
    rel_spread      DOUBLE,
    vendor_iv       DOUBLE,
    own_iv          DOUBLE,              -- NULL when no solution
    iv_reason       VARCHAR   NOT NULL,  -- ok, or why own_iv is NULL
    iv_diff         DOUBLE,              -- own_iv - vendor_iv
    crossed_market  BOOLEAN,
    zero_bid        BOOLEAN,
    wide_spread     BOOLEAN,
    stale_trade     BOOLEAN,
    low_liquidity   BOOLEAN,
    PRIMARY KEY (snapshot_ts, underlying, expiry, option_type, strike)
);
"""


# SVI fit per snapshot/underlying/expiry. rmse is in implied-vol units
# (decimal, unweighted) so it is comparable across expiries.
SVI_PARAMS_DDL = """
CREATE TABLE IF NOT EXISTS svi_params (
    snapshot_ts  TIMESTAMP NOT NULL,
    underlying   VARCHAR   NOT NULL,
    expiry       DATE      NOT NULL,
    a            DOUBLE,
    b            DOUBLE,
    rho          DOUBLE,
    m            DOUBLE,
    sigma        DOUBLE,
    rmse         DOUBLE,
    n_points     INTEGER   NOT NULL,
    fit_status   VARCHAR   NOT NULL,   -- ok / too_few_points / bad_input / ...
    PRIMARY KEY (snapshot_ts, underlying, expiry)
);
"""

# Per-quote distance from the fitted smile (input to anomaly detection).
SVI_RESIDUALS_DDL = """
CREATE TABLE IF NOT EXISTS svi_residuals (
    snapshot_ts     TIMESTAMP NOT NULL,
    underlying      VARCHAR   NOT NULL,
    expiry          DATE      NOT NULL,
    option_type     VARCHAR   NOT NULL,
    strike          DOUBLE    NOT NULL,
    log_moneyness   DOUBLE,
    tte_years       DOUBLE,
    w_market        DOUBLE,
    w_fit           DOUBLE,
    own_iv          DOUBLE,
    fit_iv          DOUBLE,
    resid_w         DOUBLE,            -- market - fit, total variance
    resid_iv        DOUBLE,            -- market - fit, implied vol
    PRIMARY KEY (snapshot_ts, underlying, expiry, option_type, strike)
);
"""


# Findings from the rule-based checks. strike / option_type are NULL when a
# finding applies to a whole expiry (or a call/put pair). is_synthetic marks
# findings produced on injected test data (Stage 6); never mix without it.
ANOMALIES_DDL = """
CREATE TABLE IF NOT EXISTS anomalies (
    snapshot_ts   TIMESTAMP NOT NULL,
    underlying    VARCHAR   NOT NULL,
    expiry        DATE      NOT NULL,
    strike        DOUBLE,
    option_type   VARCHAR,
    rule          VARCHAR   NOT NULL,
    severity      VARCHAR   NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    details       VARCHAR,
    is_synthetic  BOOLEAN   NOT NULL DEFAULT FALSE
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
    con.execute(IV_QUOTES_DDL)
    con.execute(SVI_PARAMS_DDL)
    con.execute(SVI_RESIDUALS_DDL)
    con.execute(ANOMALIES_DDL)
