"""Forward price per expiry, estimated from put-call parity.

Parity (European, discount D = exp(-rT)):   C - P = D * (F - K)
so each strike with both a call and a put gives an estimate
    F_K = K + (C_mid - P_mid) / D.
We take the median over the strikes nearest spot (robust to one bad pair).

Assumptions (state these in interviews):
- One flat risk-free rate (config.RISK_FREE_RATE) for all expiries.
- Parity is exact only for European options. These are American, so near
  ATM there is a small error (early-exercise premium on puts pushes F down
  slightly). Using near-ATM strikes and the median keeps it small.
- Dividends and borrow costs are NOT modelled; they are absorbed into F.
Fallback: F = S * exp((r - q) * T) when too few clean pairs exist.
"""
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from ivmon import config


class ForwardEstimate(NamedTuple):
    forward: float
    method: str  # "parity" or "spot_fallback"
    n_pairs: int  # parity pairs used (0 for fallback)


def spot_forward(spot: float, t: float) -> float:
    """Forward implied by spot, flat rate and dividend yield."""
    return spot * math.exp((config.RISK_FREE_RATE - config.DIVIDEND_YIELD) * t)


def usable_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Keep quotes with a defined, trustworthy mid (no excluded flags)."""
    ok = quotes["mid"].notna() & (quotes["mid"] > 0)
    for flag in config.FORWARD_EXCLUDE_FLAGS:
        ok &= ~quotes[flag].astype(bool)
    return quotes[ok]


def estimate_forward(quotes: pd.DataFrame, spot: float, t: float) -> ForwardEstimate:
    """Estimate F for one underlying/expiry.

    `quotes` needs columns: strike, option_type, mid, and the flag columns.
    """
    spot_ok = spot is not None and math.isfinite(spot) and spot > 0
    if spot_ok:
        fallback = ForwardEstimate(spot_forward(spot, t), "spot_fallback", 0)
    else:
        fallback = ForwardEstimate(float("nan"), "no_forward", 0)

    good = usable_quotes(quotes)
    calls = good[good["option_type"] == "C"].set_index("strike")["mid"]
    puts = good[good["option_type"] == "P"].set_index("strike")["mid"]
    strikes = calls.index.intersection(puts.index)
    if len(strikes) == 0 or not spot_ok:
        return fallback

    log_m = np.abs(np.log(strikes.to_numpy(dtype=float) / spot))
    near = strikes[log_m <= config.FORWARD_MAX_LOG_MONEYNESS]
    near = sorted(near, key=lambda k: abs(math.log(k / spot)))[: config.FORWARD_N_PAIRS]
    if len(near) < config.FORWARD_MIN_PAIRS:
        return fallback

    discount = math.exp(-config.RISK_FREE_RATE * t)
    estimates = [k + (calls[k] - puts[k]) / discount for k in near]
    return ForwardEstimate(float(np.median(estimates)), "parity", len(near))
