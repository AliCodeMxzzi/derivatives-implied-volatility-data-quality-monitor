"""Threshold-based alerting.

`evaluate_alerts` is pure (metrics in, alerts out) so the threshold logic is
easy to test. `send_alerts` prints in dry-run mode, or posts to a Slack
incoming webhook whose URL comes from an environment variable. No secret
ever lives in the code or the repo.
"""
import json
import logging
import os
import urllib.request
from dataclasses import dataclass

from ivmon import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    name: str
    severity: str  # "critical" or "warning"
    message: str
    value: float | None = None
    threshold: float | None = None


def default_thresholds() -> dict:
    """Read thresholds from config at call time (so tests can override)."""
    return {
        "anomaly_rate": config.ALERT_ANOMALY_RATE,
        "calibration_failure_rate": config.ALERT_CALIBRATION_FAILURE_RATE,
        "zero_bid_rate": config.ALERT_ZERO_BID_RATE,
        "min_rows_per_ticker": config.ALERT_MIN_ROWS_PER_TICKER,
    }


def evaluate_alerts(metrics: dict, thresholds: dict | None = None) -> list[Alert]:
    """Turn a metrics dict into alerts. Missing metrics (None) are not checked.

    Expected metric keys: ingest_rows (ticker -> rows), zero_bid_rate,
    calibration_failure_rate, anomaly_rate, failed_steps (list of names).
    """
    t = default_thresholds() if thresholds is None else thresholds
    alerts: list[Alert] = []

    for step in metrics.get("failed_steps") or []:
        alerts.append(Alert("pipeline_step_failed", "critical",
                            f"pipeline step '{step}' failed"))

    for ticker, rows in sorted((metrics.get("ingest_rows") or {}).items()):
        if rows < t["min_rows_per_ticker"]:
            alerts.append(Alert(
                "ingestion_gap", "critical",
                f"{ticker}: only {rows} rows ingested "
                f"(minimum {t['min_rows_per_ticker']})",
                rows, t["min_rows_per_ticker"]))

    zero_bid = metrics.get("zero_bid_rate")
    if zero_bid is not None and zero_bid > t["zero_bid_rate"]:
        alerts.append(Alert(
            "zero_bid_rate", "critical",
            f"{zero_bid:.0%} of quotes have no bid (above {t['zero_bid_rate']:.0%}): "
            "snapshot was probably pulled outside market hours",
            zero_bid, t["zero_bid_rate"]))

    cal = metrics.get("calibration_failure_rate")
    if cal is not None and cal > t["calibration_failure_rate"]:
        alerts.append(Alert(
            "calibration_failures", "warning",
            f"{cal:.0%} of expiries failed SVI calibration "
            f"(above {t['calibration_failure_rate']:.0%})",
            cal, t["calibration_failure_rate"]))

    anomaly = metrics.get("anomaly_rate")
    if anomaly is not None and anomaly > t["anomaly_rate"]:
        alerts.append(Alert(
            "anomaly_rate", "warning",
            f"{anomaly:.1%} of fitted quotes flagged by the rule-based checks "
            f"(above {t['anomaly_rate']:.0%})",
            anomaly, t["anomaly_rate"]))
    return alerts


def format_alerts(alerts: list[Alert], snapshot_ts=None) -> str:
    header = f"IV quality monitor: {len(alerts)} alert(s)"
    if snapshot_ts is not None:
        header += f" for snapshot {snapshot_ts}"
    lines = [header] + [f"[{a.severity.upper()}] {a.name}: {a.message}" for a in alerts]
    return "\n".join(lines)


def _post_to_slack(url: str, text: str) -> None:
    request = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def send_alerts(alerts: list[Alert], snapshot_ts=None, send: bool = False) -> str:
    """Dry-run prints the alert text. With send=True and the webhook env var set,
    posts to Slack. Never raises: a failed send is logged and the text is printed.

    Returns what happened: "none", "printed", "sent" or "send_failed".
    """
    if not alerts:
        log.info("no alerts")
        return "none"
    text = format_alerts(alerts, snapshot_ts)
    url = os.environ.get(config.SLACK_WEBHOOK_ENV)
    if send and url:
        try:
            _post_to_slack(url, text)
            log.info("alert sent to Slack")
            return "sent"
        except Exception as exc:  # network, bad URL, HTTP error: never crash the run
            log.error("Slack send failed: %s", exc)
            print(text)
            return "send_failed"
    if send and not url:
        log.warning("--send-alerts given but %s is not set; printing instead",
                    config.SLACK_WEBHOOK_ENV)
    print("[dry run] " + text)
    return "printed"
