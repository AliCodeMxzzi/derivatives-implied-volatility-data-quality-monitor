import logging
import math

import numpy as np
import pandas as pd
import pytest

from ivmon import checks, config
from ivmon.db import get_connection, init_schema
from ivmon.pricing import black_price
from ivmon.svi import svi_total_variance

SNAP = pd.Timestamp("2026-01-02 21:00:00")
E1 = pd.Timestamp("2026-02-20")
E2 = pd.Timestamp("2026-03-20")

GOOD = dict(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)         # min g = +0.37
BAD = dict(a=0.0001, b=1.5, rho=-0.9, m=0.0, sigma=0.05)       # min g = -4.8
BETWEEN = dict(a=0.0, b=0.2, rho=0.0, m=0.0, sigma=0.01)       # g<0 only near k=0


def params_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: (expiry, dict of svi params)."""
    return pd.DataFrame([
        {"snapshot_ts": SNAP, "underlying": "SPY", "expiry": e, **p}
        for e, p in rows])


def resid_df(expiry, svi: dict, ks, t: float = 0.1, noise=None) -> pd.DataFrame:
    """Quotes lying on (or near) the given SVI curve."""
    ks = np.asarray(ks, dtype=float)
    w_fit = svi_total_variance(ks, **svi)
    w_mkt = w_fit if noise is None else w_fit + noise
    own_iv = np.sqrt(w_mkt / t)
    fit_iv = np.sqrt(w_fit / t)
    return pd.DataFrame({
        "snapshot_ts": SNAP, "underlying": "SPY", "expiry": expiry,
        "option_type": ["P" if k < 0 else "C" for k in ks],
        "strike": 100 * np.exp(ks), "log_moneyness": ks, "tte_years": t,
        "w_market": w_mkt, "w_fit": w_fit, "own_iv": own_iv, "fit_iv": fit_iv,
        "resid_w": w_mkt - w_fit, "resid_iv": own_iv - fit_iv,
    })


KS = np.linspace(-0.3, 0.3, 13)


# ---------------------------------------------------------------- butterfly

def test_density_g_known_values() -> None:
    assert checks.svi_density_g(np.linspace(-1, 1, 500), **GOOD).min() > 0.3
    assert checks.svi_density_g(np.linspace(-1, 1, 500), **BAD).min() < -4


def test_butterfly_no_flags_for_valid_smile() -> None:
    out = checks.butterfly_check(params_df([(E1, GOOD)]), resid_df(E1, GOOD, KS))
    assert out.empty


def test_butterfly_flags_arbitrageable_smile() -> None:
    out = checks.butterfly_check(params_df([(E1, BAD)]), resid_df(E1, BAD, KS))
    assert len(out) > 0
    assert set(out["rule"]) == {"butterfly_arbitrage"}
    assert "high" in set(out["severity"])
    assert out["strike"].notna().all()


def test_butterfly_violation_between_quotes_is_expiry_level() -> None:
    ks = np.array([-0.3, -0.2, 0.2, 0.3])
    out = checks.butterfly_check(params_df([(E1, BETWEEN)]), resid_df(E1, BETWEEN, ks))
    assert len(out) == 1
    assert pd.isna(out["strike"].iloc[0])  # expiry-level finding
    assert "between quotes" in out["details"].iloc[0]


# ----------------------------------------------------------------- calendar

def test_calendar_ok_when_variance_increases() -> None:
    later = dict(GOOD, a=GOOD["a"] + 0.02)  # strictly higher total variance
    p = params_df([(E1, GOOD), (E2, later)])
    r = pd.concat([resid_df(E1, GOOD, KS, t=0.1), resid_df(E2, later, KS, t=0.2)])
    assert checks.calendar_check(p, r).empty


def test_calendar_flags_decreasing_variance() -> None:
    later = dict(GOOD, a=GOOD["a"] - 0.02)  # later expiry has LESS variance
    p = params_df([(E1, GOOD), (E2, later)])
    r = pd.concat([resid_df(E1, GOOD, KS, t=0.1), resid_df(E2, later, KS, t=0.2)])
    out = checks.calendar_check(p, r)
    assert len(out) == len(KS)
    assert set(out["rule"]) == {"calendar_arbitrage"}
    assert (out["expiry"] == E2).all()  # blamed on the later expiry
    assert "high" in set(out["severity"])  # drop is ~30% of w near ATM


def test_calendar_ignores_k_outside_earlier_range() -> None:
    later = dict(GOOD, a=GOOD["a"] - 0.02)
    p = params_df([(E1, GOOD), (E2, later)])
    narrow = np.linspace(-0.06, 0.06, 5)  # earlier expiry only covers +-6%
    r = pd.concat([resid_df(E1, GOOD, narrow, t=0.1), resid_df(E2, later, KS, t=0.2)])
    out = checks.calendar_check(p, r)
    # later quotes inside +-6%: k = -0.05, 0.0, +0.05 (KS has a 0.05 step)
    assert len(out) == 3


# ------------------------------------------------------------------- parity

FWD = 101.0
T = 0.25


def parity_inputs(break_call_by: float = 0.0):
    rows = []
    for k in [96, 98, 100, 102, 104, 106]:
        for kind in ("C", "P"):
            mid = black_price(FWD, k, T, 0.25, config.RISK_FREE_RATE, kind)
            if kind == "C" and k == 102:
                mid += break_call_by
            rows.append({
                "underlying": "SPY", "expiry": E1, "strike": float(k),
                "option_type": kind, "mid": mid, "spread": 0.04,
                "crossed_market": False, "zero_bid": False})
    quotes = pd.DataFrame(rows)
    fwd = pd.DataFrame([{
        "snapshot_ts": SNAP, "underlying": "SPY", "expiry": E1, "forward": FWD,
        "forward_method": "parity", "tte_years": T}])
    return quotes, fwd


def test_parity_clean_chain_has_no_breaks() -> None:
    assert checks.parity_check(*parity_inputs()).empty


def test_parity_break_is_flagged_at_right_strike() -> None:
    out = checks.parity_check(*parity_inputs(break_call_by=1.0))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["strike"] == 102.0
    assert row["rule"] == "put_call_parity"
    assert row["severity"] == "high"  # 1.0 vs tolerance ~0.06
    assert pd.isna(row["option_type"])  # pair-level finding


def test_parity_gap_inside_spread_is_tolerated() -> None:
    assert checks.parity_check(*parity_inputs(break_call_by=0.03)).empty


def test_parity_skipped_when_forward_is_fallback() -> None:
    quotes, fwd = parity_inputs(break_call_by=1.0)
    fwd["forward_method"] = "spot_fallback"
    assert checks.parity_check(quotes, fwd).empty


def test_parity_ignores_flagged_quotes() -> None:
    quotes, fwd = parity_inputs(break_call_by=1.0)
    quotes.loc[(quotes["strike"] == 102.0) & (quotes["option_type"] == "C"), "zero_bid"] = True
    assert checks.parity_check(quotes, fwd).empty


# ----------------------------------------------------------------- residual

def test_residual_outlier_flags_the_bad_quote_only() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.0003, size=len(KS))
    r = resid_df(E1, GOOD, KS, noise=noise)
    r.loc[6, "resid_iv"] += 0.15  # one quote 15 vol points off
    r.loc[6, "own_iv"] += 0.15
    out = checks.residual_outlier_check(r)
    assert len(out) == 1
    assert out.iloc[0]["strike"] == pytest.approx(r.loc[6, "strike"])
    assert out.iloc[0]["severity"] == "high"


def test_residual_outlier_nothing_flagged_on_clean_fit() -> None:
    rng = np.random.default_rng(1)
    r = resid_df(E1, GOOD, KS, noise=rng.normal(0, 0.0003, size=len(KS)))
    assert checks.residual_outlier_check(r).empty


def test_residual_outlier_zero_mad_does_not_crash() -> None:
    r = resid_df(E1, GOOD, KS)  # perfect fit: all residuals ~0 -> MAD ~0
    r["resid_iv"] = 0.0
    assert checks.residual_outlier_check(r).empty


def test_residual_min_abs_guard() -> None:
    # tiny MAD makes a 0.2-vol-point miss look extreme in z, but it is trivial
    r = resid_df(E1, GOOD, KS)
    r["resid_iv"] = np.where(np.arange(len(KS)) % 2 == 0, 1e-5, -1e-5)
    r.loc[6, "resid_iv"] = 0.002
    assert checks.residual_outlier_check(r).empty


# ---------------------------------------------------------------- stability

def test_stability_flags_big_move_and_ignores_small() -> None:
    now = params_df([(E1, GOOD)])
    same = params_df([(E1, GOOD)])
    moved = params_df([(E1, dict(GOOD, a=GOOD["a"] + 0.05))])
    key = ("SPY", E1)
    t = {key: 0.1}
    assert checks.stability_check(now, same, t, t).empty
    out = checks.stability_check(moved, now, t, t)
    assert len(out) == 1
    assert out.iloc[0]["rule"] == "param_stability"
    assert "a 0.040->0.090" in out.iloc[0]["details"]


def test_stability_compares_curves_not_raw_params() -> None:
    # Raw parameters differ but the curves are nearly identical -> no flag.
    base = dict(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    k = np.linspace(-0.1, 0.1, 41)
    target = svi_total_variance(k, **base)
    # shift 'a' down a hair and compensate with b: curve changes very little
    twin = dict(base, a=base["a"] - 1e-4, b=base["b"] * 1.001)
    assert np.sqrt(np.mean((target - svi_total_variance(k, **twin)) ** 2)) < 1e-3
    key = ("SPY", E1)
    out = checks.stability_check(
        params_df([(E1, twin)]), params_df([(E1, base)]), {key: 0.1}, {key: 0.1})
    assert out.empty


# ------------------------------------------------------ orchestration (DB)

def _seed_db(con, with_bad_curve: bool = False) -> None:
    """Minimal DB state that run_checks reads: params, residuals, quotes, iv."""
    svi = BAD if with_bad_curve else GOOD
    p = params_df([(E1, svi)]).assign(rmse=0.001, n_points=len(KS), fit_status="ok")
    r = resid_df(E1, svi, KS)
    con.register("p_in", p)
    con.register("r_in", r)
    con.execute("INSERT INTO svi_params (snapshot_ts, underlying, expiry, a, b, rho, m, "
                "sigma, rmse, n_points, fit_status) SELECT snapshot_ts, underlying, "
                "expiry, a, b, rho, m, sigma, rmse, n_points, fit_status FROM p_in").fetchall()
    con.execute("INSERT INTO svi_residuals SELECT snapshot_ts, underlying, expiry, "
                "option_type, strike, log_moneyness, tte_years, w_market, w_fit, own_iv, "
                "fit_iv, resid_w, resid_iv FROM r_in").fetchall()


def test_run_checks_stores_findings_and_skips_stability(caplog) -> None:
    con = get_connection(":memory:")
    init_schema(con)
    _seed_db(con, with_bad_curve=True)
    # run_checks also reads clean_quotes / iv_quotes (empty here is fine)
    con.execute("CREATE TABLE clean_quotes AS SELECT * FROM (SELECT "
                "CAST(NULL AS TIMESTAMP) AS snapshot_ts, '' AS underlying, "
                "CAST(NULL AS DATE) AS expiry) WHERE FALSE").fetchall()
    with caplog.at_level(logging.INFO):
        findings = checks.run_checks(con)
    assert any("param_stability SKIPPED" in m for m in caplog.messages)
    assert len(findings) > 0
    stored = con.execute("SELECT rule, is_synthetic FROM anomalies").df()
    assert len(stored) == len(findings)
    assert set(stored["rule"]) == {"butterfly_arbitrage"}
    assert not stored["is_synthetic"].any()
    assert not (findings["rule"] == "param_stability").any()


def test_run_checks_is_idempotent_and_keeps_synthetic_rows() -> None:
    con = get_connection(":memory:")
    init_schema(con)
    _seed_db(con, with_bad_curve=True)
    con.execute("CREATE TABLE clean_quotes AS SELECT * FROM (SELECT "
                "CAST(NULL AS TIMESTAMP) AS snapshot_ts, '' AS underlying, "
                "CAST(NULL AS DATE) AS expiry) WHERE FALSE").fetchall()
    con.execute("INSERT INTO anomalies VALUES ('2026-01-02 21:00:00', 'SPY', "
                "'2026-02-20', 100, 'C', 'test', 'low', 'x', TRUE)").fetchall()
    n1 = len(checks.run_checks(con))
    n2 = len(checks.run_checks(con))
    real = con.execute("SELECT count(*) FROM anomalies WHERE NOT is_synthetic").fetchall()[0][0]
    synth = con.execute("SELECT count(*) FROM anomalies WHERE is_synthetic").fetchall()[0][0]
    assert n1 == n2 == real
    assert synth == 1  # synthetic row never deleted by a real run
