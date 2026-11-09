import math

import numpy as np
import pandas as pd
import pytest

from ivmon import config, ml
from ivmon.checks import run_checks


def normal_features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "rel_resid": rng.normal(0, 0.01, n),
        "rel_spread": rng.uniform(0.02, 0.15, n),
        "log_moneyness": rng.uniform(-0.2, 0.2, n),
        "log_tte": rng.uniform(-4, -1, n),
        "log_vol_oi": rng.normal(-1, 0.5, n),
        "vendor_gap": rng.normal(0.02, 0.01, n),
    })


def test_build_features_shape_and_values(model_con) -> None:
    con, ts = model_con
    f = ml.build_features(con, ts)
    assert list(f.columns) == ml.KEY + ml.FEATURES
    assert len(f) == 21  # one OTM quote per strike
    assert not f[ml.FEATURES].isna().any().any()
    # volume 10, open interest 100 -> log(11/101)
    assert f["log_vol_oi"].iloc[0] == pytest.approx(math.log(11 / 101))
    # clean model data fits tightly: residual is a tiny share of the vol
    assert f["rel_resid"].abs().max() < 0.05


def test_outliers_that_are_off_on_several_features_are_flagged() -> None:
    # A genuinely bad quote is usually odd on several features at once:
    # big residual, wide spread, big gap to the vendor's IV.
    f = normal_features()
    outliers = pd.DataFrame({
        "rel_resid": [0.5, -0.5, 0.6, -0.4, 0.45, 0.55],
        "rel_spread": [0.9, 1.1, 0.8, 1.2, 1.0, 0.95],
        "log_moneyness": [0.0] * 6, "log_tte": [-3.0] * 6, "log_vol_oi": [-1.0] * 6,
        "vendor_gap": [0.3, -0.3, 0.35, -0.25, 0.28, 0.32],
    })
    data = pd.concat([f, outliers], ignore_index=True)
    out = ml.score_quotes(data, contamination=0.05)
    planted = out.iloc[-6:]
    assert planted["ml_flag"].sum() >= 5
    # flag rate is the contamination setting, give or take rounding
    assert out["ml_flag"].mean() == pytest.approx(0.05, abs=0.01)


def test_single_feature_outlier_is_diluted_but_still_scores_high() -> None:
    """Known limitation: the forest picks features at random, so an outlier
    that is extreme in ONE of six features is easy to miss at a fixed flag
    rate. It still scores above the typical quote."""
    f = normal_features()
    outliers = pd.DataFrame({
        "rel_resid": [0.5, -0.5, 0.6, -0.4, 0.45, 0.55],
        "rel_spread": [0.05] * 6, "log_moneyness": [0.0] * 6,
        "log_tte": [-3.0] * 6, "log_vol_oi": [-1.0] * 6, "vendor_gap": [0.02] * 6,
    })
    out = ml.score_quotes(pd.concat([f, outliers], ignore_index=True), contamination=0.05)
    assert out.iloc[-6:]["ml_score"].median() > out.iloc[:-6]["ml_score"].median()


def test_scoring_is_reproducible() -> None:
    f = normal_features()
    a = ml.score_quotes(f, seed=3)["ml_flag"].tolist()
    b = ml.score_quotes(f, seed=3)["ml_flag"].tolist()
    assert a == b


def test_too_few_rows_skips_model() -> None:
    out = ml.score_quotes(normal_features(n=10))
    assert not out["ml_flag"].any()
    assert out["ml_score"].isna().all()


def test_severity_bands() -> None:
    assert ml._severity(0.9, 0.8, 0.7) == "high"
    assert ml._severity(0.75, 0.8, 0.7) == "medium"
    assert ml._severity(0.6, 0.8, 0.7) == "low"


def test_run_ml_stores_findings_idempotently_and_survives_rule_rerun(
        model_con, monkeypatch) -> None:
    monkeypatch.setattr(config, "ML_MIN_ROWS", 10)
    con, ts = model_con
    flagged = ml.run_ml(con, ts)
    assert len(flagged) >= 1

    def stored() -> int:
        return con.execute(
            "SELECT count(*) FROM anomalies WHERE rule = ? AND NOT is_synthetic",
            [ml.ML_RULE]).fetchall()[0][0]

    assert stored() == len(flagged)
    ml.run_ml(con, ts)
    assert stored() == len(flagged)  # re-run replaces, does not duplicate
    run_checks(con, ts)
    assert stored() == len(flagged)  # rule-based re-run leaves ML rows alone
