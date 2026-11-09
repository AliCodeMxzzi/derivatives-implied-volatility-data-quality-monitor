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

# SVI calibration (Stage 4).
SVI_MIN_POINTS: int = 8  # fewer solved quotes than this -> skip the expiry
SVI_MIN_REL_SPREAD: float = 0.02  # floor on relative spread when weighting
SVI_ROBUST_FRACTION: float = 0.05  # soft_l1 scale = this * median(w)
PLOT_DIR = DATA_DIR / "plots"

# Rule-based anomaly checks (Stage 5).
BUTTERFLY_TOL: float = 1e-6  # flag when g(k) < -tol
BUTTERFLY_HIGH: float = -0.5  # g(k) below this -> severity high (else medium)
CALENDAR_TOL_REL: float = 1e-4  # tolerated relative drop in total variance
CALENDAR_MEDIUM_REL: float = 0.02  # relative drop >= this -> medium
CALENDAR_HIGH_REL: float = 0.10  # relative drop >= this -> high
PARITY_TOL_MULT: float = 1.5  # allowed |C-P-D(F-K)| = mult * avg(C,P spread)
PARITY_MIN_ABS: float = 0.01  # but never tighter than one cent
PARITY_MAX_LOG_MONEYNESS: float = 0.10  # American early exercise grows away from ATM
PARITY_MEDIUM_RATIO: float = 2.0  # excess / tolerance >= this -> medium
PARITY_HIGH_RATIO: float = 5.0  # ... >= this -> high
RESID_Z_THRESHOLD: float = 3.5  # robust z (Iglewicz-Hoaglin) above this -> flag
RESID_MIN_ABS_IV: float = 0.01  # and |residual| >= 1 vol point (guards tiny MAD)
RESID_MEDIUM_Z: float = 6.0
RESID_HIGH_Z: float = 10.0
STABILITY_GRID_K: float = 0.10  # compare smiles for k in [-this, +this]
STABILITY_RMS_IV: float = 0.03  # RMS smile change above 3 vol pts -> flag
STABILITY_HIGH_RMS_IV: float = 0.08

# ML anomaly layer (Stage 6).
ML_CONTAMINATION: float = 0.05  # share of quotes the forest flags BY CONSTRUCTION
ML_N_ESTIMATORS: int = 200
ML_SEED: int = 42
ML_MIN_ROWS: int = 50  # fewer scored quotes than this -> skip the model
ML_SEVERITY_HIGH_Q: float = 0.99  # score quantile for severity high
ML_SEVERITY_MEDIUM_Q: float = 0.975  # ... for medium (flagged below -> low)

# Synthetic anomaly injection (Stage 6). Magnitudes are my choices and
# directly influence the recall numbers; they are not calibrated to reality.
SYNTH_SEED: int = 7
SYNTH_N_PER_TYPE: int = 25
SYNTH_DB_PATH = DATA_DIR / "synthetic.duckdb"
SYNTH_SPIKE_FACTORS: tuple[float, ...] = (2.5, 0.4)  # bid/ask multiplied
SYNTH_STALE_SHIFTS: tuple[float, ...] = (0.85, 1.15)  # bid/ask multiplied
SYNTH_STALE_DAYS: int = 10  # last_trade_ts moved back this far
SYNTH_IV_FACTORS: tuple[float, ...] = (1.3, 0.7)  # implied vol multiplied

# Remediation policy (Stage 7): rule -> severity -> action.
#   quarantine = exclude the quote from the surface
#   correct    = publish the fitted-surface value instead (clearly marked)
#   escalate   = change nothing; a human / quant must look
#   none       = no action (finding stays in `anomalies`)
# Reasoning, so the choices can be defended:
# - Only residual outliers are quote-level problems we can act on ourselves.
#   High severity (very far from the fit) -> correct; medium -> quarantine.
# - Structural findings (arbitrage, parity, stability) point at the model or
#   the data feed, not one quote: escalate, never auto-edit.
# - ML findings only escalate at high severity: Stage 6 showed they are near
#   chance on injected errors, so they must not edit or exclude data.
# - crossed_market is a Stage 2 flag; a crossed quote is unusable: quarantine.
REMEDIATION_POLICY: dict[str, dict[str, str]] = {
    "residual_outlier": {"high": "correct", "medium": "quarantine", "low": "none"},
    "crossed_market": {"high": "quarantine", "medium": "quarantine", "low": "quarantine"},
    "butterfly_arbitrage": {"high": "escalate", "medium": "escalate", "low": "none"},
    "calendar_arbitrage": {"high": "escalate", "medium": "escalate", "low": "none"},
    "put_call_parity": {"high": "escalate", "medium": "escalate", "low": "none"},
    "param_stability": {"high": "escalate", "medium": "escalate", "low": "none"},
    "ml_isolation_forest": {"high": "escalate", "medium": "none", "low": "none"},
}
REMEDIATION_DEFAULT_ACTION: str = "escalate"  # for any rule not listed above

# Daily report (Stage 8).
REPORT_DIR = DATA_DIR / "reports"
REPORT_TOP_N: int = 10  # anomalies listed in the report
TREND_SNAPSHOTS: int = 7  # prior snapshots shown in the trend table

# Alert thresholds (Stage 8). An alert fires when a value is STRICTLY above
# its threshold (or, for row counts, below the minimum). These are initial,
# round-number settings chosen before any history existed; they are not
# calibrated. Revisit them once weeks of snapshots have accumulated.
ALERT_ANOMALY_RATE: float = 0.10  # flagged quotes / fitted quotes
ALERT_CALIBRATION_FAILURE_RATE: float = 0.10  # failed fits / expiries
ALERT_ZERO_BID_RATE: float = 0.90  # share of zero-bid quotes: likely off-hours pull
ALERT_MIN_ROWS_PER_TICKER: int = 50  # fewer ingested rows -> ingestion gap
SLACK_WEBHOOK_ENV: str = "SLACK_WEBHOOK_URL"  # name of the env var; never the URL

# Forward estimation from put-call parity (Stage 3).
FORWARD_N_PAIRS: int = 5  # use the N strikes closest to spot
FORWARD_MIN_PAIRS: int = 2  # fewer usable pairs -> fall back to spot
FORWARD_MAX_LOG_MONEYNESS: float = 0.10  # only strikes within ~10% of spot
# A parity pair needs both legs free of these flags:
FORWARD_EXCLUDE_FLAGS: list[str] = ["crossed_market", "zero_bid", "wide_spread"]
