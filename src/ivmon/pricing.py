"""Option pricing and implied volatility.

We price in forward form (Black-76):
    call = D * (F * N(d1) - K * N(d2))
    put  = D * (K * N(-d2) - F * N(-d1))
with D = exp(-r T), d1 = (ln(F/K) + 0.5 s^2) / s, d2 = d1 - s, s = sigma sqrt(T).
This equals Black-Scholes with dividend yield q when F = S * exp((r - q) T).
"""
import math
from typing import NamedTuple

from scipy.optimize import brentq

from ivmon import config


class IVResult(NamedTuple):
    iv: float  # NaN when no solution
    reason: str  # "ok" or why there is no solution


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def black_price(
    forward: float, strike: float, t: float, sigma: float, rate: float,
    option_type: str,
) -> float:
    """Price a European option from the forward. option_type is 'C' or 'P'."""
    discount = math.exp(-rate * t)
    if option_type == "C":
        intrinsic = max(forward - strike, 0.0)
    else:
        intrinsic = max(strike - forward, 0.0)
    if sigma <= 0.0 or t <= 0.0:
        return discount * intrinsic

    s = sigma * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if option_type == "C":
        return discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    return discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))


def bs_price(
    spot: float, strike: float, t: float, rate: float, div_yield: float,
    sigma: float, option_type: str,
) -> float:
    """Black-Scholes price with continuous dividend yield q."""
    forward = spot * math.exp((rate - div_yield) * t)
    return black_price(forward, strike, t, sigma, rate, option_type)


def implied_vol(
    price: float, forward: float, strike: float, t: float, rate: float,
    option_type: str,
) -> IVResult:
    """Solve for sigma with Brent's method. Never raises; returns a reason.

    Reasons: ok, bad_input, non_positive_price, below_intrinsic,
    above_upper_bound, below_min_vol, above_max_vol, solver_failed.
    """
    nan = float("nan")
    values = (price, forward, strike, t, rate)
    if not all(math.isfinite(v) for v in values) or option_type not in ("C", "P"):
        return IVResult(nan, "bad_input")
    if forward <= 0 or strike <= 0 or t <= 0:
        return IVResult(nan, "bad_input")
    if price <= 0:
        return IVResult(nan, "non_positive_price")

    discount = math.exp(-rate * t)
    if option_type == "C":
        intrinsic = discount * max(forward - strike, 0.0)
        upper = discount * forward
    else:
        intrinsic = discount * max(strike - forward, 0.0)
        upper = discount * strike
    if price <= intrinsic:
        return IVResult(nan, "below_intrinsic")  # no time value: no positive vol fits
    if price >= upper:
        return IVResult(nan, "above_upper_bound")  # no-arbitrage ceiling

    def gap(sigma: float) -> float:
        return black_price(forward, strike, t, sigma, rate, option_type) - price

    lo, hi = config.IV_MIN_VOL, config.IV_MAX_VOL
    if gap(lo) > 0:
        return IVResult(nan, "below_min_vol")
    if gap(hi) < 0:
        return IVResult(nan, "above_max_vol")
    try:
        sigma = brentq(gap, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    except (RuntimeError, ValueError):
        return IVResult(nan, "solver_failed")
    return IVResult(sigma, "ok")
