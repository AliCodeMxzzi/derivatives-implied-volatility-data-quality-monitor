import logging

import pytest

from ivmon import alert, config

HEALTHY = {
    "ingest_rows": {"SPY": 1600, "AAPL": 650},
    "zero_bid_rate": 0.2,
    "calibration_failure_rate": 0.0,
    "anomaly_rate": 0.05,
    "failed_steps": [],
}


def names(alerts) -> list[str]:
    return [a.name for a in alerts]


def test_healthy_metrics_raise_no_alerts() -> None:
    assert alert.evaluate_alerts(HEALTHY) == []


def test_missing_metrics_are_not_checked() -> None:
    assert alert.evaluate_alerts({"failed_steps": [], "ingest_rows": {}}) == []


@pytest.mark.parametrize("key, name, threshold_key", [
    ("anomaly_rate", "anomaly_rate", "anomaly_rate"),
    ("calibration_failure_rate", "calibration_failures", "calibration_failure_rate"),
    ("zero_bid_rate", "zero_bid_rate", "zero_bid_rate"),
])
def test_rate_alerts_fire_only_strictly_above_threshold(key, name, threshold_key) -> None:
    limit = alert.default_thresholds()[threshold_key]
    at_limit = {**HEALTHY, key: limit}
    above = {**HEALTHY, key: limit + 0.01}
    assert name not in names(alert.evaluate_alerts(at_limit))  # equal -> no alert
    fired = alert.evaluate_alerts(above)
    assert names(fired) == [name]
    assert fired[0].value == pytest.approx(limit + 0.01)
    assert fired[0].threshold == limit


def test_ingestion_gap_per_ticker() -> None:
    metrics = {**HEALTHY, "ingest_rows": {"SPY": 1600, "XOM": 3, "GLD": 0}}
    fired = alert.evaluate_alerts(metrics)
    assert names(fired) == ["ingestion_gap", "ingestion_gap"]
    assert {"GLD", "XOM"} == {a.message.split(":")[0] for a in fired}
    assert all(a.severity == "critical" for a in fired)


def test_ingestion_at_exact_minimum_is_fine() -> None:
    minimum = config.ALERT_MIN_ROWS_PER_TICKER
    assert alert.evaluate_alerts({**HEALTHY, "ingest_rows": {"SPY": minimum}}) == []
    assert names(alert.evaluate_alerts({**HEALTHY, "ingest_rows": {"SPY": minimum - 1}})) == ["ingestion_gap"]


def test_failed_steps_are_critical_alerts() -> None:
    fired = alert.evaluate_alerts({**HEALTHY, "failed_steps": ["calibrate", "checks"]})
    assert names(fired) == ["pipeline_step_failed", "pipeline_step_failed"]
    assert all(a.severity == "critical" for a in fired)


def test_thresholds_can_be_overridden() -> None:
    strict = {**alert.default_thresholds(), "anomaly_rate": 0.01}
    assert names(alert.evaluate_alerts(HEALTHY, strict)) == ["anomaly_rate"]


def test_default_thresholds_follow_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALERT_ANOMALY_RATE", 0.5)
    assert alert.evaluate_alerts({**HEALTHY, "anomaly_rate": 0.4}) == []


def test_multiple_problems_all_reported() -> None:
    bad = {**HEALTHY, "zero_bid_rate": 0.99, "anomaly_rate": 0.5,
           "ingest_rows": {"SPY": 0}, "failed_steps": ["ingest"]}
    assert set(names(alert.evaluate_alerts(bad))) == {
        "pipeline_step_failed", "ingestion_gap", "zero_bid_rate", "anomaly_rate"}


# ------------------------------------------------------------------ sending

SAMPLE = [alert.Alert("anomaly_rate", "warning", "too many", 0.2, 0.1)]


def test_no_alerts_means_nothing_sent(capsys) -> None:
    assert alert.send_alerts([]) == "none"
    assert capsys.readouterr().out == ""


def test_dry_run_prints_and_never_touches_the_network(monkeypatch, capsys) -> None:
    monkeypatch.setenv(config.SLACK_WEBHOOK_ENV, "https://example.invalid/hook")
    monkeypatch.setattr(alert, "_post_to_slack",
                        lambda *a, **k: pytest.fail("network used in dry run"))
    assert alert.send_alerts(SAMPLE, send=False) == "printed"
    out = capsys.readouterr().out
    assert "[dry run]" in out and "anomaly_rate" in out


def test_send_without_webhook_falls_back_to_printing(monkeypatch, capsys, caplog) -> None:
    monkeypatch.delenv(config.SLACK_WEBHOOK_ENV, raising=False)
    with caplog.at_level(logging.WARNING):
        assert alert.send_alerts(SAMPLE, send=True) == "printed"
    assert "not set" in caplog.text
    assert "[dry run]" in capsys.readouterr().out


def test_send_posts_to_the_webhook_from_the_environment(monkeypatch) -> None:
    sent = {}
    monkeypatch.setenv(config.SLACK_WEBHOOK_ENV, "https://example.invalid/hook")
    monkeypatch.setattr(alert, "_post_to_slack", lambda url, text: sent.update(url=url, text=text))
    assert alert.send_alerts(SAMPLE, snapshot_ts="2026-01-02", send=True) == "sent"
    assert sent["url"] == "https://example.invalid/hook"
    assert "anomaly_rate" in sent["text"] and "2026-01-02" in sent["text"]


def test_failed_send_never_raises(monkeypatch, capsys) -> None:
    def boom(url, text):
        raise OSError("network down")

    monkeypatch.setenv(config.SLACK_WEBHOOK_ENV, "https://example.invalid/hook")
    monkeypatch.setattr(alert, "_post_to_slack", boom)
    assert alert.send_alerts(SAMPLE, send=True) == "send_failed"
    assert "anomaly_rate" in capsys.readouterr().out  # alert text is not lost


def test_no_secret_is_hardcoded() -> None:
    from pathlib import Path
    source = Path(alert.__file__).read_text()
    assert "hooks.slack.com" not in source
