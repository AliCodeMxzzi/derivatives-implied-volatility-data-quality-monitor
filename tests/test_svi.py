import numpy as np
import pandas as pd
import pytest

from ivmon import config
from ivmon.db import get_connection, init_schema
from ivmon.svi import build_svi, fit_svi, point_weights, svi_total_variance

TRUE = dict(a=0.02, b=0.15, rho=-0.4, m=0.02, sigma=0.15)
T = 0.25
K = np.linspace(-0.4, 0.4, 41)


def true_w(k: np.ndarray = K) -> np.ndarray:
    return svi_total_variance(k, **TRUE)


def test_svi_formula_known_point() -> None:
    # With m=0, rho=0: w(0) = a + b*sigma
    assert svi_total_variance(np.array([0.0]), 0.01, 0.2, 0.0, 0.0, 0.1)[0] == \
        pytest.approx(0.01 + 0.2 * 0.1)


def test_recovers_parameters_noise_free() -> None:
    fit = fit_svi(K, true_w(), T)
    assert fit.status == "ok"
    for name, value in TRUE.items():
        assert getattr(fit, name) == pytest.approx(value, abs=2e-3), name
    assert fit.rmse < 1e-4


def test_recovers_curve_with_noise() -> None:
    rng = np.random.default_rng(42)
    noisy = true_w() + rng.normal(0, 0.0005, size=len(K))
    fit = fit_svi(K, noisy, T)
    assert fit.status == "ok"
    fitted = svi_total_variance(K, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
    # Fitted curve stays close to the TRUE curve (not just to the noisy data)
    assert np.sqrt(np.mean((fitted - true_w()) ** 2)) < 0.0007
    assert fit.rho < 0  # skew direction recovered


def test_robust_loss_resists_one_bad_quote() -> None:
    w = true_w().copy()
    w[20] *= 2.5  # one wildly wrong quote at the money
    fit = fit_svi(K, w, T)
    fitted = svi_total_variance(K, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
    away = np.arange(len(K)) != 20
    assert np.sqrt(np.mean((fitted[away] - true_w()[away]) ** 2)) < 0.003
    assert fitted[20] < w[20] * 0.8  # curve did NOT chase the outlier


def test_too_few_points_is_skipped() -> None:
    fit = fit_svi(K[:5], true_w()[:5], T)
    assert fit.status == "too_few_points"
    assert np.isnan(fit.a)
    assert fit.n_points == 5


@pytest.mark.parametrize("bad_w", [0.0, -0.1, float("nan")])
def test_bad_input_is_rejected(bad_w) -> None:
    w = true_w().copy()
    w[3] = bad_w
    assert fit_svi(K, w, T).status == "bad_input"


def test_bad_time_is_rejected() -> None:
    assert fit_svi(K, true_w(), 0.0).status == "bad_input"


def test_heavier_weights_pull_the_fit() -> None:
    # Right half of the data is shifted up; weighting it more moves the fit up there.
    w = true_w().copy()
    w[K > 0.1] += 0.02
    even = fit_svi(K, w, T)
    favour_right = np.where(K > 0.1, 50.0, 1.0)
    right = fit_svi(K, w, T, favour_right)
    idx = np.argmax(K)
    f_even = svi_total_variance(K[idx], even.a, even.b, even.rho, even.m, even.sigma)
    f_right = svi_total_variance(K[idx], right.a, right.b, right.rho, right.m, right.sigma)
    assert abs(f_right - w[idx]) < abs(f_even - w[idx])


def test_point_weights_floor_and_missing() -> None:
    s = pd.Series([0.001, 0.1, np.nan])
    w = point_weights(s)
    assert w[0] == pytest.approx(1 / config.SVI_MIN_REL_SPREAD)  # floored
    assert w[1] == pytest.approx(10.0)
    assert w[2] == pytest.approx(1.0)  # missing spread -> low weight


# ---- end to end through the database -------------------------------------

SNAP = "2026-01-02 21:00:00"


def add_iv_row(con, expiry: str, strike: float, own_iv: float, forward: float, t: float,
               option_type: str = "C", underlying: str = "SPY") -> None:
    con.execute(
        "INSERT INTO iv_quotes (snapshot_ts, underlying, expiry, option_type, strike, "
        "spot, forward, forward_method, tte_years, mid, rel_spread, vendor_iv, own_iv, "
        "iv_reason, iv_diff) VALUES "
        f"('{SNAP}', '{underlying}', '{expiry}', '{option_type}', {strike}, 100, "
        f"{forward}, 'parity', {t}, 1.0, 0.05, 0.2, {own_iv}, 'ok', 0)"
    ).fetchall()


def test_build_svi_end_to_end() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    forward = 100.0
    strikes = forward * np.exp(K)
    iv = np.sqrt(true_w() / T)
    for s, v in zip(strikes, iv):
        add_iv_row(con, "2026-04-03", float(s), float(v), forward, T)
    # second expiry with too few points
    for s, v in list(zip(strikes, iv))[:4]:
        add_iv_row(con, "2026-05-01", float(s), float(v), forward, 0.3)

    n_ok, n_total = build_svi(con)
    assert (n_ok, n_total) == (1, 2)

    params = con.execute("SELECT * FROM svi_params ORDER BY expiry").df()
    assert params["fit_status"].tolist() == ["ok", "too_few_points"]
    assert params["n_points"].tolist() == [41, 4]
    assert params["rmse"].iloc[0] < 1e-4
    assert pd.isna(params["a"].iloc[1])

    resid = con.execute("SELECT * FROM svi_residuals").df()
    assert len(resid) == 41  # skipped expiry has no residuals
    assert resid["resid_iv"].abs().max() < 1e-3
    assert resid["w_market"].tolist() == pytest.approx((resid["w_fit"] + resid["resid_w"]).tolist())


def test_build_svi_is_idempotent() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    forward = 100.0
    for s, v in zip(forward * np.exp(K), np.sqrt(true_w() / T)):
        add_iv_row(con, "2026-04-03", float(s), float(v), forward, T)
    build_svi(con)
    build_svi(con)
    assert con.execute("SELECT count(*) FROM svi_params").fetchall()[0][0] == 1
    assert con.execute("SELECT count(*) FROM svi_residuals").fetchall()[0][0] == 41
