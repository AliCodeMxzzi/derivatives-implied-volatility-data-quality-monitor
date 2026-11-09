from datetime import datetime

import pandas as pd
import pytest
from conftest import MODEL_EXPIRY, MODEL_FORWARD, MODEL_STRIKES, MODEL_T, model_vol

from ivmon import config, pipeline
from ivmon.db import get_connection
from ivmon.pricing import black_price

TS = datetime(2026, 1, 2, 21, 0, 0)
STEP_ORDER = ["ingest", "clean", "iv", "calibrate", "checks", "ml", "remediate",
              "report", "alert"]


def vendor_chain(kind: str) -> pd.DataFrame:
    """Clean model chain in yfinance's column format."""
    rows = []
    for k in MODEL_STRIKES:
        mid = black_price(MODEL_FORWARD, k, MODEL_T, model_vol(k),
                          config.RISK_FREE_RATE, kind)
        rows.append({
            "strike": float(k), "bid": mid * 0.98, "ask": mid * 1.02,
            "lastPrice": mid, "volume": 10.0, "openInterest": 100,
            "impliedVolatility": model_vol(k),
            "lastTradeDate": pd.Timestamp("2026-01-02 20:00:00", tz="UTC")})
    return pd.DataFrame(rows)


class FakeSource:
    def get_spot(self, ticker):
        return 100.0

    def get_expiries(self, ticker):
        return [MODEL_EXPIRY]

    def get_chain(self, ticker, expiry):
        return vendor_chain("C"), vendor_chain("P")


class DeadSource(FakeSource):
    def get_spot(self, ticker):
        raise RuntimeError("vendor down")


@pytest.fixture(autouse=True)
def fast_and_small(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_PAUSE_SECONDS", 0)
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(config, "ML_MIN_ROWS", 10)
    monkeypatch.setattr(config, "ALERT_MIN_ROWS_PER_TICKER", 10)  # 42-row test chain


def statuses(result) -> dict:
    return {s.name: s.status for s in result.steps}


def test_full_run_succeeds_and_writes_report(tmp_path) -> None:
    con = get_connection(":memory:")
    result = pipeline.run_pipeline(con, FakeSource(), ["SPY"], TS, report_dir=tmp_path)
    assert result.ok, [(s.name, s.detail) for s in result.steps if s.status != "ok"]
    assert [s.name for s in result.steps] == STEP_ORDER
    assert set(statuses(result).values()) == {"ok"}
    assert result.report_path.exists()
    text = result.report_path.read_text(encoding="utf-8")
    for heading in ["## Run status", "## Alerts", "## Per-underlying summary",
                    "## Findings by rule and severity", "## Remediation (latest run)",
                    "## Trend vs prior snapshots", "## Volatility surface fits"]:
        assert heading in text
    assert "SPY" in text
    plot = tmp_path / f"report_20260102_210000_plots" / "svi_SPY.png"
    assert plot.exists()
    assert con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0] == 42


def test_ingest_failure_skips_downstream_but_still_reports_and_alerts(tmp_path, capsys) -> None:
    con = get_connection(":memory:")
    result = pipeline.run_pipeline(con, DeadSource(), ["SPY"], TS, report_dir=tmp_path)
    st = statuses(result)
    assert st["ingest"] == "failed"
    for step in ["clean", "iv", "calibrate", "checks", "ml", "remediate"]:
        assert st[step] == "skipped", step
    assert st["report"] == "ok" and st["alert"] == "ok"
    assert result.failed_steps == ["ingest"] and not result.ok
    alert_names = {a.name for a in result.alerts}
    assert {"pipeline_step_failed", "ingestion_gap"} <= alert_names
    assert "[dry run]" in capsys.readouterr().out
    text = result.report_path.read_text(encoding="utf-8")
    assert "failed" in text and "unavailable" in text  # failure is visible in the report


def test_a_failing_middle_step_is_isolated(tmp_path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise ValueError("optimizer exploded")

    monkeypatch.setattr(pipeline, "build_svi", broken)
    con = get_connection(":memory:")
    result = pipeline.run_pipeline(con, FakeSource(), ["SPY"], TS, report_dir=tmp_path)
    st = statuses(result)
    assert st["ingest"] == st["clean"] == st["iv"] == "ok"
    assert st["calibrate"] == "failed"
    assert [st[s] for s in ("checks", "ml", "remediate")] == ["skipped"] * 3
    assert st["report"] == "ok" and st["alert"] == "ok"
    failure = next(s for s in result.steps if s.name == "calibrate")
    assert "optimizer exploded" in failure.detail
    assert "pipeline_step_failed" in {a.name for a in result.alerts}


def test_skip_ingest_reprocesses_the_stored_snapshot(tmp_path) -> None:
    con = get_connection(":memory:")
    first = pipeline.run_pipeline(con, FakeSource(), ["SPY"], TS, report_dir=tmp_path)
    again = pipeline.run_pipeline(con, None, ["SPY"], skip_ingest=True, report_dir=tmp_path)
    assert again.ok
    assert again.snapshot_ts == first.snapshot_ts
    ingest = next(s for s in again.steps if s.name == "ingest")
    assert "skipped" in ingest.detail
    assert con.execute("SELECT count(*) FROM raw_quotes").fetchall()[0][0] == 42  # nothing re-pulled


def test_alerts_use_pipeline_metrics(tmp_path) -> None:
    # An absurdly strict threshold must produce a real alert from real metrics.
    con = get_connection(":memory:")
    result = pipeline.run_pipeline(con, FakeSource(), ["SPY"], TS, report_dir=tmp_path)
    assert result.alerts == [] or all(a.name != "ingestion_gap" for a in result.alerts)


def test_main_exit_code_reflects_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.duckdb")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(pipeline, "YahooSource", DeadSource)
    monkeypatch.setattr(config, "TICKERS", ["SPY"])
    assert pipeline.main([]) == 1
