# Methodology

## Notation
- `F` forward for an expiry, `K` strike, `T` time to expiry in years (ACT/365, expiry at 21:00 UTC).
- Log-moneyness `k = ln(K / F)`.
- Total variance `w(k) = sigma_imp(k)^2 * T`.

## Quote cleaning
Quotes are flagged, never deleted: zero bid, wide spread, stale last trade,
low liquidity, crossed market. Mid = (bid + ask) / 2; relative spread =
(ask - bid) / mid. Raw data stays untouched so every downstream number can be
traced back.

## Forward estimation
Per expiry, put-call parity gives `F = K + e^{rT} (C - P)` at each strike. We
take the median over the 5 strikes nearest the money (robust to one bad
quote). If parity cannot be computed (too few two-sided pairs) we fall back to
spot (with the configured rate/dividend assumption), which is less accurate.
Result on the live snapshot: parity forward for most expiries, with fallback
for the rest.

## Implied volatility
Black-76 price in forward form, inverted with Brent's method on a bounded
vol interval (bracketing, guaranteed convergence, no derivative needed, unlike
Newton which can diverge at low vega). Failures return NaN plus a reason code
(zero bid, below intrinsic, non-positive price, no bracket). Only OTM options
are inverted: OTM prices are mostly time value, so the American early-exercise
premium that Black-76 ignores is small there.

## SVI smile
Raw SVI per expiry:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

- `a` vertical level, `b` slope of the wings, `rho` skew (tilt), `m` horizontal
  shift of the smile minimum, `sigma` smoothness of the vertex.
- Total variance is used because it is close to linear in the wings and makes
  the calendar condition simple (w must not decrease with T).
- One fit per expiry because the smile shape differs sharply across maturities.
- Fitted with `scipy.optimize.least_squares`, `soft_l1` loss (so one bad quote
  does not drag the whole curve), 4 starting points, weights `1 / relative
  spread` (tight, liquid quotes count more). Expiries with too few points are
  skipped and logged.

## Rule-based checks
1. **Butterfly arbitrage**: the Gatheral density function `g(k) >= 0`. Negative
   `g` means negative implied risk-neutral density, i.e. a butterfly spread with
   a negative price. It is checked on the fitted curve, which is why it can flag
   the fit as much as the data.
2. **Calendar arbitrage**: at the same log-moneyness, total variance must not
   fall as expiry increases. A violation means a longer option is cheaper
   than a shorter one in variance terms.
3. **Put-call parity**: `C - P` must match the forward-implied value within the
   quote spreads. The tolerance is spread-aware so wide quotes do not trigger
   false findings.
4. **Residual outliers**: residual of a quote's IV against the fitted curve,
   scored with a robust z (median and MAD), flagged beyond a threshold.
5. **Parameter stability**: today's fitted curve vs the previous snapshot's,
   compared as curves (not raw parameters, which trade off against each other).
   Skipped when no previous fit exists.

Real data violates these slightly because of discrete tick sizes, stale quotes,
non-synchronous bid/ask, and American exercise.

## ML layer
Pooled Isolation Forest on six features: SVI residual, relative spread,
log-moneyness, time to expiry, volume / open interest ratio, vendor-vs-own IV
gap. Pooled because per-underlying samples are small. Contamination is fixed at
5%, so the flag rate is a setting, not a result.

## Evaluation
Known corruptions (price spike, stale quote, swapped bid/ask, strike typo, IV
shift) are injected into a copy of a real snapshot, stored in an isolated
database with `is_synthetic = TRUE`. Detectors are scored on precision and
recall. This measures sensitivity to those corruptions only.

## Remediation
A config policy maps rule and severity to `quarantine` (exclude from the
surface), `correct` (replace with the fitted value, marked), or `escalate`
(human review; the default for anything unlisted). Every action goes to an
append-only `remediation_log` with run id, before and after values, and
trigger rules. `curated_quotes` is a view applying the latest run to the
untouched raw data. Fit quality is compared like-for-like on quotes untouched
by remediation.
