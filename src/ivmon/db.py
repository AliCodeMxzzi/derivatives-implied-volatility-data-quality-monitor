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


# Append-only audit trail. Code only ever INSERTs here. A new remediation run
# gets a new run_id; older runs stay untouched, and curated_quotes uses the
# latest run per snapshot.
REMEDIATION_LOG_DDL = """
CREATE TABLE IF NOT EXISTS remediation_log (
    run_id        INTEGER   NOT NULL,
    run_ts        TIMESTAMP NOT NULL,
    snapshot_ts   TIMESTAMP NOT NULL,
    underlying    VARCHAR   NOT NULL,
    expiry        DATE      NOT NULL,
    strike        DOUBLE,              -- NULL for expiry-level findings
    option_type   VARCHAR,             -- NULL for expiry- or pair-level findings
    action        VARCHAR   NOT NULL CHECK (action IN ('quarantine', 'correct', 'escalate')),
    rule          VARCHAR   NOT NULL,  -- triggering rule(s), '; '-joined
    severity      VARCHAR   NOT NULL,  -- highest severity among triggers
    reason        VARCHAR,             -- finding details + policy applied
    before_bid    DOUBLE,
    before_ask    DOUBLE,
    before_mid    DOUBLE,
    after_mid     DOUBLE,              -- only for 'correct': the published value
    is_synthetic  BOOLEAN   NOT NULL DEFAULT FALSE
);
"""

# One row per remediation run, even a run with zero actions, so "latest run"
# is always well defined. Also append-only.
REMEDIATION_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS remediation_runs (
    run_id        INTEGER   NOT NULL PRIMARY KEY,
    run_ts        TIMESTAMP NOT NULL,
    snapshot_ts   TIMESTAMP NOT NULL,
    is_synthetic  BOOLEAN   NOT NULL,
    n_findings    INTEGER   NOT NULL,
    n_actions     INTEGER   NOT NULL
);
"""

# Surface refit WITHOUT quarantined / corrected quotes (same columns as svi_params).
SVI_PARAMS_CURATED_DDL = SVI_PARAMS_DDL.replace("svi_params", "svi_params_curated")


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
    con.execute(REMEDIATION_LOG_DDL)
    con.execute(REMEDIATION_RUNS_DDL)
    con.execute(SVI_PARAMS_CURATED_DDL)
