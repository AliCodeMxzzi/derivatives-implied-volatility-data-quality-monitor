"""Central configuration: tickers, paths, and thresholds live here only.

Later stages add their thresholds to this file so every rule is tunable
and reviewable in one place.
"""
from pathlib import Path

# Repo root = two levels above src/ivmon/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ivmon.duckdb"

# Equities / ETFs, then commodity proxies (commodity ETF options stand in
# for commodity derivatives).
EQUITY_TICKERS: list[str] = ["SPY", "AAPL", "NVDA", "XOM"]
COMMODITY_PROXY_TICKERS: list[str] = ["GLD", "USO", "SLV"]
TICKERS: list[str] = EQUITY_TICKERS + COMMODITY_PROXY_TICKERS

# Ingestion settings (Stage 1).
MAX_EXPIRIES: int = 6  # nearest N expiries per ticker; None would mean all
MAX_RETRIES: int = 3  # attempts per vendor call before giving up
RETRY_BACKOFF_SECONDS: float = 2.0  # wait grows: 2s, 4s, ...
REQUEST_PAUSE_SECONDS: float = 0.5  # polite pause between vendor calls

# Pricing assumptions (used from Stage 3 onward).
RISK_FREE_RATE: float = 0.04  # simplifying assumption, flat rate
DIVIDEND_YIELD: float = 0.0  # simplifying assumption, continuous yield q
