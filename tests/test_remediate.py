from pathlib import Path

import pandas as pd
import pytest

from ivmon import config, remediate
from ivmon.pricing import black_price
from ivmon.svi import build_svi

EXPIRY = pd.Timestamp("2026-04-03")


def add_finding(con, ts, strike, kind, rule, severity, details="test finding",
                synthetic=False) -> None:
    """Insert one anomaly. strike/kind may be None (expiry- or pair-level)."""
    con.execute(
        "INSERT INTO anomalies VALUES (?, 'SPY', ?, ?, ?, ?, ?, ?, ?)",
        [ts, EXPIRY, strike, kind, rule, severity, details, synthetic]).fetchall()


def findings_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"underlying": "SPY", "expiry": EXPIRY, "strike": s, "option_type": t,
          "rule": r, "severity": sev, "details": "d"} for s, t, r, sev in rows])


# ------------------------------------------------------------------- policy

def test_policy_covers_every_severity_with_a_valid_action() -> None:
    valid = {"quarantine", "correct", "escalate", "none"}
    for rule, by_sev in config.REMEDIATION_POLICY.items():
        assert set(by_sev) == {"low", "medium", "high"}, rule
        assert set(by_sev.values()) <= valid, rule


def test_unknown_rule_defaults_to_escalate() -> None:
    assert remediate.policy_action("some_new_rule", "high") == "escalate"


def test_ml_findings_never_edit_data() -> None:
    for sev, action in config.REMEDIATION_POLICY["ml_isolation_forest"].items():
        assert action in ("escalate", "none"), sev


def test_decide_actions_per_policy() -> None:
    f = findings_frame([
        (90.0, "P", "residual_outlier", "high"),       # -> correct
        (92.0, "P", "residual_outlier", "medium"),     # -> quarantine
        (94.0, "P", "residual_outlier", "low"),        # -> no action
        (96.0, None, "put_call_parity", "medium"),     # pair-level -> escalate
        (None, None, "butterfly_arbitrage", "high"),   # expiry-level -> escalate
        (104.0, "C", "ml_isolation_forest", "high"),   # -> escalate
        (106.0, "C", "ml_isolation_forest", "low"),    # -> no action
    ])
    a = remediate.decide_actions(f)
    # pandas stores a missing option_type as NaN; normalise to None for lookups
    by = {(r.strike, None if pd.isna(r.option_type) else r.option_type): r.action
          for r in a.itertuples()}
    assert by[(90.0, "P")] == "correct"
    assert by[(92.0, "P")] == "quarantine"
    assert (94.0, "P") not in by
    assert by[(96.0, None)] == "escalate"
    assert by[(104.0, "C")] == "escalate"
    assert (106.0, "C") not in by
    assert len(a) == 5
    assert a[a["strike"].isna()]["action"].tolist() == ["escalate"]  # expiry-level


def test_conflicts_resolve_to_most_conservative_money_action() -> None:
    f = findings_frame([
        (98.0, "P", "residual_outlier", "high"),     # correct ...
        (98.0, "P", "crossed_market", "high"),       # ... but quarantine wins
        (102.0, "C", "residual_outlier", "high"),    # correct, plus
        (102.0, "C", "butterfly_arbitrage", "high"), # escalate (both kept)
    ])
    a = remediate.decide_actions(f)
    p98 = a[a["strike"] == 98.0]
    assert p98["action"].tolist() == ["quarantine"]
    assert set(p98.iloc[0]["rule"].split("; ")) == {"residual_outlier", "crossed_market"}
    c102 = a[a["strike"] == 102.0]
    assert sorted(c102["action"]) == ["correct", "escalate"]


def test_no_findings_means_no_actions() -> None:
    assert remediate.decide_actions(findings_frame([])).empty


# --------------------------------------------------- log, view, audit trail

@pytest.fixture
def remediated(model_con):
    con, ts = model_con
    add_finding(con, ts, 90.0, "P", "residual_outlier", "high")
    add_finding(con, ts, 92.0, "P", "residual_outlier", "medium")
    add_finding(con, ts, 94.0, "P", "residual_outlier", "low")
    add_finding(con, ts, 96.0, None, "put_call_parity", "medium")
    add_finding(con, ts, None, None, "butterfly_arbitrage", "high")
    add_finding(con, ts, 104.0, "C", "ml_isolation_forest", "high")
    rows = remediate.apply_remediation(con, ts)
    return con, ts, rows


def test_log_is_complete_and_matches_policy(remediated) -> None:
    con, ts, rows = remediated
    log = con.execute("SELECT * FROM remediation_log").df()
    assert len(log) == len(rows) == 5  # low-severity finding takes no action
    assert log["run_id"].eq(1).all()
    for col in ["run_ts", "snapshot_ts", "rule", "severity", "reason", "action"]:
        assert log[col].notna().all(), col
    assert set(log["action"]) == {"correct", "quarantine", "escalate"}
    assert (log["reason"].str.contains("policy:")).all()
    assert not log["is_synthetic"].any()


def test_log_records_before_and_after_values(remediated) -> None:
    con, ts, _ = remediated
    log = con.execute("SELECT * FROM remediation_log WHERE strike = 90 AND option_type = 'P'").df().iloc[0]
    clean = con.execute("SELECT * FROM clean_quotes WHERE strike = 90 AND option_type = 'P'").df().iloc[0]
    assert (log["before_bid"], log["before_ask"], log["before_mid"]) == pytest.approx(
        (clean["bid"], clean["ask"], clean["mid"]))
    # corrected value = price at the fitted-surface IV
    resid = con.execute("SELECT fit_iv FROM svi_residuals WHERE strike = 90 AND option_type = 'P'").df().iloc[0]
    iv = con.execute("SELECT forward, tte_years FROM iv_quotes WHERE strike = 90 AND option_type = 'P'").df().iloc[0]
    expected = black_price(iv["forward"], 90.0, iv["tte_years"], resid["fit_iv"],
                           config.RISK_FREE_RATE, "P")
    assert log["action"] == "correct"
    assert log["after_mid"] == pytest.approx(expected)
    quarantined = con.execute("SELECT after_mid FROM remediation_log WHERE action = 'quarantine'").df()
    assert quarantined["after_mid"].isna().all()  # quarantine publishes no value


def test_curated_view_applies_actions_and_leaves_raw_alone(remediated) -> None:
    con, ts, _ = remediated
    raw_before = con.execute("SELECT sum(bid), sum(ask), count(*) FROM raw_quotes").fetchall()
    view = con.execute("SELECT * FROM curated_quotes").df()
    clean_n = con.execute("SELECT count(*) FROM clean_quotes").fetchall()[0][0]
    assert len(view) == clean_n  # nothing is dropped from the view

    def row(strike, kind):
        return view[(view["strike"] == strike) & (view["option_type"] == kind)].iloc[0]

    corrected = row(90.0, "P")
    assert corrected["is_corrected"] and not corrected["is_quarantined"]
    assert corrected["curated_mid"] != pytest.approx(corrected["raw_mid"])
    assert row(92.0, "P")["is_quarantined"]
    assert row(104.0, "C")["is_escalated"]
    untouched = row(80.0, "P")
    assert untouched["curated_mid"] == pytest.approx(untouched["raw_mid"])
    assert not (untouched["is_quarantined"] or untouched["is_corrected"] or untouched["is_escalated"])
    assert con.execute("SELECT sum(bid), sum(ask), count(*) FROM raw_quotes").fetchall() == raw_before


def test_log_is_append_only_across_runs(remediated) -> None:
    con, ts, _ = remediated
    first = con.execute("SELECT * FROM remediation_log WHERE run_id = 1 ORDER BY ALL").df()
    remediate.apply_remediation(con, ts)
    again = con.execute("SELECT * FROM remediation_log ORDER BY ALL").df()
    assert set(again["run_id"]) == {1, 2}
    assert len(again) == 2 * len(first)
    pd.testing.assert_frame_equal(
        again[again["run_id"] == 1].reset_index(drop=True), first.reset_index(drop=True))
    assert con.execute("SELECT count(*) FROM remediation_runs").fetchall()[0][0] == 2


def test_remediation_code_never_updates_or_deletes_the_log() -> None:
    source = Path(remediate.__file__).read_text()
    for table in ("remediation_log", "remediation_runs"):
        assert f"UPDATE {table}" not in source
        assert f"DELETE FROM {table}" not in source


def test_latest_run_wins_even_if_it_takes_no_actions(remediated) -> None:
    con, ts, _ = remediated
    con.execute("DELETE FROM anomalies").fetchall()  # findings resolved upstream
    remediate.apply_remediation(con, ts)  # run 2: zero actions
    view = con.execute("SELECT count(*) FILTER (WHERE is_quarantined OR is_corrected OR is_escalated) "
                       "FROM curated_quotes").fetchall()[0][0]
    assert view == 0  # stale run-1 actions are not shown
    assert con.execute("SELECT n_actions FROM remediation_runs WHERE run_id = 2").fetchall()[0][0] == 0


def test_synthetic_and_real_findings_are_kept_apart(model_con) -> None:
    con, ts = model_con
    add_finding(con, ts, 92.0, "P", "residual_outlier", "medium", synthetic=True)
    assert remediate.apply_remediation(con, ts, synthetic=False).empty
    rows = remediate.apply_remediation(con, ts, synthetic=True)
    assert len(rows) == 1 and rows["is_synthetic"].all()


def test_crossed_market_flag_is_quarantined(model_con) -> None:
    con, ts = model_con
    con.execute("UPDATE clean_quotes SET crossed_market = TRUE "
                "WHERE strike = 82 AND option_type = 'C'").fetchall()
    rows = remediate.apply_remediation(con, ts)
    assert rows["action"].tolist() == ["quarantine"]
    assert rows.iloc[0]["rule"] == "crossed_market"


# -------------------------------------------------------------------- refit

def test_refit_excludes_acted_quotes_and_improves_fit(model_con) -> None:
    con, ts = model_con
    # Corrupt one OTM quote's IV, refit so the surface is bent by it
    con.execute("UPDATE iv_quotes SET own_iv = own_iv * 1.8 "
                "WHERE strike = 100 AND option_type = 'P'").fetchall()
    build_svi(con, ts)
    add_finding(con, ts, 100.0, "P", "residual_outlier", "medium")   # quarantine
    add_finding(con, ts, 110.0, "C", "residual_outlier", "high")     # correct
    remediate.apply_remediation(con, ts)
    params = remediate.refit_curated(con, ts)

    assert params["n_points"].iloc[0] == 21 - 2  # both acted-on quotes left out
    result = remediate.compare_fit(con, ts)
    o = result["overall"]
    assert o["n_quotes"] == 19
    assert o["rmse_after"] < o["rmse_before"]
    # originals are untouched: the pre-remediation fit is still in svi_params
    assert con.execute("SELECT n_points FROM svi_params").fetchall()[0][0] == 21


def test_compare_fit_uses_one_common_quote_set(model_con) -> None:
    con, ts = model_con
    add_finding(con, ts, 92.0, "P", "residual_outlier", "medium")
    remediate.apply_remediation(con, ts)
    remediate.refit_curated(con, ts)
    # Excluding extra keys (e.g. injected quotes) shrinks the common set
    extra = {("SPY", EXPIRY, "P", 94.0)}
    base = remediate.compare_fit(con, ts)["overall"]["n_quotes"]
    fewer = remediate.compare_fit(con, ts, exclude_keys=extra)["overall"]["n_quotes"]
    assert base == 20 and fewer == 19
