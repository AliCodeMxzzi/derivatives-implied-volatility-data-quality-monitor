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

# Cleaning thresholds (Stage 2). Flags only; nothing is deleted.
WIDE_SPREAD_REL: float = 0.50  # (ask-bid)/mid above this -> wide_spread
STALE_TRADE_DAYS: float = 4.0  # last trade older than this -> stale_trade
#   4 days so a normal weekend (Fri trade seen on Mon = 3 days) is not stale,
#   while a trade from last week is.

# Time-to-expiry convention (Stage 2).
# Expiry is treated as the 4pm New York close, approximated as 21:00 UTC
# (true in winter; one hour early in summer, which is negligible).
# Year fraction = actual seconds / (365 * 86400), i.e. ACT/365 Fixed.
EXPIRY_CLOSE_UTC_HOUR: int = 21
DAYS_PER_YEAR: float = 365.0

# Pricing assumptions (Stage 3).
RISK_FREE_RATE: float = 0.04  # simplifying assumption: one flat rate for all expiries
DIVIDEND_YIELD: float = 0.0  # simplifying assumption: continuous yield q.
#   Only used in the spot-based forward fallback; when the forward comes
#   from put-call parity, dividends and borrow cost are absorbed into it.

# IV solver bounds (annualised vol). Prices outside this range -> NaN + reason.
IV_MIN_VOL: float = 1e-4
IV_MAX_VOL: float = 5.0

# Quote flags that make a mid unusable for IV. These rows get a NaN IV with
# reason "skipped_<flag>" instead of a garbage number.
IV_SKIP_FLAGS: list[str] = ["crossed_market", "zero_bid"]

# Forward estimation from put-call parity (Stage 3).
FORWARD_N_PAIRS: int = 5  # use the N strikes closest to spot
FORWARD_MIN_PAIRS: int = 2  # fewer usable pairs -> fall back to spot
FORWARD_MAX_LOG_MONEYNESS: float = 0.10  # only strikes within ~10% of spot
# A parity pair needs both legs free of these flags:
FORWARD_EXCLUDE_FLAGS: list[str] = ["crossed_market", "zero_bid", "wide_spread"]
