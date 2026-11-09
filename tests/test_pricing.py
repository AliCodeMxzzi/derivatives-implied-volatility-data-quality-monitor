import math

import pytest

from ivmon import config
from ivmon.pricing import black_price, bs_price, implied_vol

R = 0.05


def test_known_black_scholes_values() -> None:
    # Textbook: S=100, K=100, T=1, r=5%, q=0, sigma=20%
    call = bs_price(100, 100, 1.0, R, 0.0, 0.2, "C")
    put = bs_price(100, 100, 1.0, R, 0.0, 0.2, "P")
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_holds_in_pricing() -> None:
    f, k, t, sigma = 103.0, 98.0, 0.5, 0.3
    c = black_price(f, k, t, sigma, R, "C")
    p = black_price(f, k, t, sigma, R, "P")
    assert c - p == pytest.approx(math.exp(-R * t) * (f - k), abs=1e-10)


def test_dividend_yield_lowers_call_price() -> None:
    no_div = bs_price(100, 100, 1.0, R, 0.0, 0.2, "C")
    with_div = bs_price(100, 100, 1.0, R, 0.03, 0.2, "C")
    assert with_div < no_div


@pytest.mark.parametrize("option_type", ["C", "P"])
@pytest.mark.parametrize("strike", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("t", [0.02, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_round_trip_price_to_iv_to_price(option_type, strike, t, sigma) -> None:
    f = 100.0
    price = black_price(f, strike, t, sigma, R, option_type)
    result = implied_vol(price, f, strike, t, R, option_type)
    if result.reason == "ok":
        # Whenever we return a vol, it must reprice to the same price.
        assert black_price(f, strike, t, result.iv, R, option_type) == \
            pytest.approx(price, abs=1e-9)
    else:
        # Only legitimate failures: the price carries no vol information.
        # Deep ITM with no time value -> below_intrinsic; deep OTM short-dated
        # options underflow to a price of exactly 0.0 -> non_positive_price.
        assert result.reason in {
            "below_intrinsic", "below_min_vol", "above_upper_bound",
            "non_positive_price",
        }
        assert math.isnan(result.iv)


def test_round_trip_recovers_vol_for_normal_options() -> None:
    price = black_price(100, 105, 0.5, 0.27, R, "C")
    result = implied_vol(price, 100, 105, 0.5, R, "C")
    assert result.reason == "ok"
    assert result.iv == pytest.approx(0.27, abs=1e-8)


def test_deep_otm_call_recovers_vol() -> None:
    price = black_price(100, 150, 0.25, 0.4, R, "C")
    result = implied_vol(price, 100, 150, 0.25, R, "C")
    assert result.reason == "ok"
    assert result.iv == pytest.approx(0.4, abs=1e-6)


def test_deep_otm_tiny_price_reprices_or_fails_cleanly() -> None:
    price = 1e-12
    result = implied_vol(price, 100, 300, 0.05, R, "C")
    if result.reason == "ok":
        assert black_price(100, 300, 0.05, result.iv, R, "C") == pytest.approx(price, abs=1e-9)
    else:
        assert math.isnan(result.iv)


def test_deep_itm_option_solves() -> None:
    price = black_price(100, 70, 1.0, 0.3, R, "C")
    result = implied_vol(price, 100, 70, 1.0, R, "C")
    assert result.reason == "ok"
    assert result.iv == pytest.approx(0.3, abs=1e-6)


def test_price_below_intrinsic_returns_nan_with_reason() -> None:
    # Forward 100, strike 80: discounted intrinsic ~ 19.2. Price 10 is impossible.
    result = implied_vol(10.0, 100, 80, 1.0, R, "C")
    assert math.isnan(result.iv)
    assert result.reason == "below_intrinsic"


def test_put_below_intrinsic() -> None:
    result = implied_vol(5.0, 100, 120, 1.0, R, "P")
    assert result.reason == "below_intrinsic"


def test_price_above_no_arbitrage_ceiling() -> None:
    # A call can never be worth more than the discounted forward.
    result = implied_vol(100.0, 100, 100, 1.0, R, "C")
    assert result.reason == "above_upper_bound"


def test_vol_outside_solver_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "IV_MAX_VOL", 0.5)
    price = black_price(100, 100, 1.0, 1.0, R, "C")  # true vol 100% > 50% cap
    assert implied_vol(price, 100, 100, 1.0, R, "C").reason == "above_max_vol"


@pytest.mark.parametrize("bad", [
    dict(price=float("nan")), dict(forward=0.0), dict(strike=-5.0), dict(t=0.0),
    dict(t=-0.1), dict(option_type="X"),
])
def test_bad_inputs_never_raise(bad) -> None:
    args = dict(price=2.0, forward=100.0, strike=100.0, t=0.5, rate=R, option_type="C")
    args.update(bad)
    result = implied_vol(**args)
    assert math.isnan(result.iv)
    assert result.reason == "bad_input"


def test_non_positive_price() -> None:
    assert implied_vol(0.0, 100, 120, 0.5, R, "C").reason == "non_positive_price"
    assert implied_vol(-1.0, 100, 120, 0.5, R, "C").reason == "non_positive_price"
