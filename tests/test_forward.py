import math

import pandas as pd
import pytest

from ivmon import config
from ivmon.forward import estimate_forward, spot_forward
from ivmon.pricing import black_price

T = 0.25
SPOT = 100.0
TRUE_F = 101.0


def chain(strikes, forward=TRUE_F, sigma=0.25, flags=None) -> pd.DataFrame:
    """Parity-consistent call/put mids built from a known forward."""
    rows = []
    for k in strikes:
        for kind in ("C", "P"):
            row = {
                "strike": float(k), "option_type": kind,
                "mid": black_price(forward, k, T, sigma, config.RISK_FREE_RATE, kind),
                "crossed_market": False, "zero_bid": False, "wide_spread": False,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_recovers_known_forward() -> None:
    est = estimate_forward(chain([92, 96, 100, 104, 108]), SPOT, T)
    assert est.method == "parity"
    assert est.n_pairs == 5
    assert est.forward == pytest.approx(TRUE_F, abs=1e-9)


def test_uses_median_so_one_bad_pair_is_ignored() -> None:
    df = chain([92, 96, 100, 104, 108])
    bad = (df["strike"] == 100.0) & (df["option_type"] == "C")
    df.loc[bad, "mid"] += 3.0  # corrupt one call
    est = estimate_forward(df, SPOT, T)
    assert est.forward == pytest.approx(TRUE_F, abs=1e-9)


def test_falls_back_to_spot_when_too_few_pairs() -> None:
    est = estimate_forward(chain([100]), SPOT, T)
    assert est.method == "spot_fallback"
    assert est.forward == pytest.approx(spot_forward(SPOT, T))
    assert est.n_pairs == 0


def test_falls_back_when_only_calls_exist() -> None:
    df = chain([96, 100, 104])
    df = df[df["option_type"] == "C"]
    assert estimate_forward(df, SPOT, T).method == "spot_fallback"


def test_flagged_quotes_are_excluded() -> None:
    df = chain([96, 100, 104])
    df.loc[df["strike"].isin([96, 100]), "zero_bid"] = True  # leaves 1 clean pair
    assert estimate_forward(df, SPOT, T).method == "spot_fallback"


def test_far_otm_strikes_ignored() -> None:
    # 70 and 130 are ~30% from spot: outside the near-ATM window
    assert estimate_forward(chain([70, 130]), SPOT, T).method == "spot_fallback"


def test_no_spot_and_no_pairs_gives_nan() -> None:
    est = estimate_forward(chain([100]), float("nan"), T)
    assert est.method == "no_forward"
    assert math.isnan(est.forward)


def test_spot_forward_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RISK_FREE_RATE", 0.04)
    monkeypatch.setattr(config, "DIVIDEND_YIELD", 0.01)
    assert spot_forward(100.0, 1.0) == pytest.approx(100.0 * math.exp(0.03))
