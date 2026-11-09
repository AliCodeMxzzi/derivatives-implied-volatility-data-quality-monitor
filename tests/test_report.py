import pandas as pd
import pytest

from ivmon import report
from ivmon.clean import build_clean_quotes
from ivmon.db import get_connection, init_schema

EXPIRY = pd.Timestamp("2026-04-03")


def test_md_table_formats_rows_and_edge_cases() -> None:
    df = pd.DataFrame({"name": ["a", "b"], "x": [1.23456789, float("nan")]})
    lines = report.md_table(df).splitlines()
    assert lines[0] == "| name | x |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| a | 1.235 |"
    assert lines[3] == "| b |  |"
    assert report.md_table(pd.DataFrame()) == "_none_"
    # dates print as dates, snapshot timestamps keep their time of day
    assert report.fmt(pd.Timestamp("2026-04-03")) == "2026-04-03"
    assert report.fmt(pd.Timestamp("2026-11-09 20:22:53")) == "2026-11-09 20:22:53"
    assert "unavailable" in report.md_table(None)


def test_metrics_from_database(model_con) -> None:
    con, ts = model_con
    con.execute("INSERT INTO anomalies VALUES (?, 'SPY', ?, 90, 'P', 'residual_outlier', "
                "'high', 'x', FALSE)", [ts, EXPIRY]).fetchall()
    con.execute("INSERT INTO anomalies VALUES (?, 'SPY', ?, 92, 'P', 'residual_outlier', "
                "'low', 'x', FALSE)", [ts, EXPIRY]).fetchall()
    con.execute("INSERT INTO anomalies VALUES (?, 'SPY', ?, NULL, NULL, 'butterfly_arbitrage', "
                "'high', 'expiry-level', FALSE)", [ts, EXPIRY]).fetchall()   # no quote: not counted
    con.execute("INSERT INTO anomalies VALUES (?, 'SPY', ?, 94, 'P', 'ml_isolation_forest', "
                "'low', 'ml', FALSE)", [ts, EXPIRY]).fetchall()              # ML: not counted
    con.execute("INSERT INTO anomalies VALUES (?, 'SPY', ?, 96, 'P', 'residual_outlier', "
                "'low', 'x', TRUE)", [ts, EXPIRY]).fetchall()                # synthetic: not counted
    # extra expiries: one failed fit, one deliberate skip
    for expiry, status in (("2026-05-01", "fit_failed"), ("2026-06-01", "too_few_points")):
        con.execute("INSERT INTO svi_params (snapshot_ts, underlying, expiry, n_points, "
                    "fit_status) VALUES (?, 'SPY', ?, 3, ?)", [ts, expiry, status]).fetchall()

    m = report.collect_metrics(con, ts, {"SPY": 42}, ["calibrate"])
    assert m["n_raw_quotes"] == 42
    assert m["n_fitted_quotes"] == 21
    assert m["n_flagged_quotes"] == 2
    assert m["anomaly_rate"] == pytest.approx(2 / 21)
    # 3 expiries, 1 real failure (too_few_points is a skip, not a failure)
    assert m["calibration_failure_rate"] == pytest.approx(1 / 3)
    assert m["zero_bid_rate"] == 0.0
    assert m["ingest_rows"] == {"SPY": 42}
    assert m["failed_steps"] == ["calibrate"]


def test_metrics_are_none_when_tables_are_missing() -> None:
    con = get_connection(":memory:")
    init_schema(con)  # no clean_quotes, no data
    m = report.collect_metrics(con, pd.Timestamp("2026-01-02"), None, [])
    assert m["zero_bid_rate"] is None
    assert m["calibration_failure_rate"] is None
    assert m["anomaly_rate"] is None


def test_trend_lists_snapshots_newest_first(model_con) -> None:
    con, ts = model_con
    one = report.trend(con, 7)
    assert len(one) == 1  # only the processed snapshot so far
    # a second ingested (but unprocessed) snapshot appears with clean-quote rates
    con.execute("INSERT INTO raw_quotes (snapshot_ts, underlying, spot, expiry, option_type, "
                "strike, bid, ask, volume, open_interest, last_trade_ts) VALUES "
                "('2026-01-03 21:00:00', 'SPY', 100, '2026-04-03', 'C', 100, 0, 0, 0, 0, "
                "'2026-01-03 20:00:00')").fetchall()
    build_clean_quotes(con)
    two = report.trend(con, 7)
    assert len(two) == 2
    assert two["snapshot_ts"].is_monotonic_decreasing
    newest = two.iloc[0]
    assert newest["zero_bid_pct"] == 100.0
    assert newest["fits_ok"] == 0  # never processed
    assert two.iloc[1]["fits_ok"] == 1


def test_report_on_empty_database_does_not_crash(tmp_path) -> None:
    con = get_connection(":memory:")
    init_schema(con)
    path = report.write_report(con, pd.Timestamp("2026-01-02 21:00:00"),
                               report.collect_metrics(con, pd.Timestamp("2026-01-02 21:00:00"), {}, []),
                               [], [], None, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "unavailable" in text
    assert "No fitted surfaces to plot" in text
