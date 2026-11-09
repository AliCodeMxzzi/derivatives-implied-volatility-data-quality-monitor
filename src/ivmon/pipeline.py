"""Daily pipeline: ingest -> clean -> IV -> calibrate -> check -> ML ->
remediate -> report -> alert.

Failure handling: every step is timed and wrapped. If a step fails, its
dependents are skipped (never run on missing or stale input), but the report
and alert steps still run so the failure itself is reported.
Exit status is non-zero if any step failed.
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from ivmon import config
from ivmon.alert import Alert, evaluate_alerts, send_alerts
from ivmon.checks import run_checks
from ivmon.clean import build_clean_quotes
from ivmon.db import get_connection, init_schema
from ivmon.ingest import OptionSource, YahooSource, collect_snapshot
from ivmon.iv import build_iv_quotes
from ivmon.ml import run_ml
from ivmon.remediate import apply_remediation, compare_fit, refit_curated
from ivmon.report import collect_metrics, write_report
from ivmon.svi import build_svi

log = logging.getLogger(__name__)

# step name -> steps that must have succeeded first
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "clean": ("ingest",),
    "iv": ("clean",),
    "calibrate": ("iv",),
    "checks": ("calibrate",),
    "ml": ("calibrate",),
    "remediate": ("checks",),
    "report": (),  # always attempted: it reports failures too
    "alert": ("report",),
}


@dataclass
class StepResult:
    name: str
    status: str  # ok / failed / skipped
    seconds: float = 0.0
    detail: str = ""


@dataclass
class PipelineResult:
    snapshot_ts: datetime
    steps: list[StepResult] = field(default_factory=list)
    report_path: Path | None = None
    alerts: list[Alert] = field(default_factory=list)
    alert_outcome: str = "none"

    @property
    def failed_steps(self) -> list[str]:
        return [s.name for s in self.steps if s.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed_steps


def _run_step(name: str, func, results: list[StepResult]) -> object:
    """Run one step with timing, logging and failure isolation."""
    by_name = {r.name: r for r in results}
    blocked = [d for d in DEPENDENCIES[name]
               if by_name.get(d) is None or by_name[d].status != "ok"]
    if blocked:
        log.warning("step %s skipped: %s did not succeed", name, ", ".join(blocked))
        results.append(StepResult(name, "skipped", 0.0, f"needs {', '.join(blocked)}"))
        return None
    start = time.perf_counter()
    try:
        value = func()
    except Exception as exc:
        seconds = time.perf_counter() - start
        log.exception("step %s FAILED", name)
        results.append(StepResult(name, "failed", seconds, f"{type(exc).__name__}: {exc}"))
        return None
    seconds = time.perf_counter() - start
    detail = value if isinstance(value, str) else ""
    log.info("step %s ok (%.1fs) %s", name, seconds, detail)
    results.append(StepResult(name, "ok", seconds, detail))
    return value


def _rows_per_ticker(con: duckdb.DuckDBPyConnection, ts: datetime, tickers: list[str]) -> dict[str, int]:
    found = dict(con.execute(
        "SELECT underlying, count(*) FROM raw_quotes WHERE snapshot_ts = ? GROUP BY 1",
        [ts]).fetchall())
    return {t: int(found.get(t, 0)) for t in tickers}


def run_pipeline(
    con: duckdb.DuckDBPyConnection, source: OptionSource | None = None,
    tickers: list[str] | None = None, snapshot_ts: datetime | None = None,
    skip_ingest: bool = False, report_dir: Path | str | None = None,
    send: bool = False,
) -> PipelineResult:
    """Run every step for one snapshot and return what happened."""
    tickers = list(config.TICKERS if tickers is None else tickers)
    if skip_ingest and snapshot_ts is None:
        snapshot_ts = con.execute("SELECT max(snapshot_ts) FROM raw_quotes").fetchall()[0][0]
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    init_schema(con)
    result = PipelineResult(snapshot_ts=snapshot_ts)
    ctx: dict = {"ingest_rows": {}, "refit": None, "alerts": []}
    steps = result.steps

    def ingest() -> str:
        if skip_ingest:
            ctx["ingest_rows"] = _rows_per_ticker(con, snapshot_ts, tickers)
            return "skipped by option; using stored snapshot"
        counts = collect_snapshot(con, source or YahooSource(), tickers, snapshot_ts)
        ctx["ingest_rows"] = {t: counts.get(t, 0) for t in tickers}
        total = sum(ctx["ingest_rows"].values())
        if total == 0:
            raise RuntimeError("no rows ingested for any ticker")
        return f"{total} rows"

    def clean() -> str:
        return f"{build_clean_quotes(con)} rows"

    def iv() -> str:
        return f"{build_iv_quotes(con, snapshot_ts)} OTM rows"

    def calibrate() -> str:
        ok, total = build_svi(con, snapshot_ts)
        return f"{ok}/{total} expiries fitted"

    def checks() -> str:
        return f"{len(run_checks(con, snapshot_ts))} findings"

    def ml() -> str:
        return f"{len(run_ml(con, snapshot_ts))} flagged"

    def remediate() -> str:
        actions = apply_remediation(con, snapshot_ts)
        refit_curated(con, snapshot_ts)
        ctx["refit"] = compare_fit(con, snapshot_ts)["overall"]
        return f"{len(actions)} actions"

    def report() -> str:
        failed = [s.name for s in steps if s.status == "failed"]
        metrics = collect_metrics(con, snapshot_ts, ctx["ingest_rows"], failed)
        ctx["alerts"] = evaluate_alerts(metrics)
        # The report step's own row is not yet in `steps`, so list it as running.
        shown = steps + [StepResult("report", "ok", 0.0, "this report")]
        refit = ctx["refit"] if ctx["refit"] and ctx["refit"]["n_quotes"] else None
        result.report_path = write_report(con, snapshot_ts, metrics, shown,
                                          ctx["alerts"], refit, report_dir)
        return str(result.report_path)

    def alert() -> str:
        result.alerts = ctx["alerts"]
        result.alert_outcome = send_alerts(result.alerts, snapshot_ts, send=send)
        return f"{len(result.alerts)} alerts ({result.alert_outcome})"

    for name, func in [("ingest", ingest), ("clean", clean), ("iv", iv),
                       ("calibrate", calibrate), ("checks", checks), ("ml", ml),
                       ("remediate", remediate), ("report", report), ("alert", alert)]:
        _run_step(name, func, steps)
    return result


def summarize(result: PipelineResult) -> str:
    lines = [f"Pipeline run for snapshot {result.snapshot_ts}:"]
    for s in result.steps:
        lines.append(f"  {s.name:<10} {s.status:<8} {s.seconds:6.1f}s  {s.detail}")
    if result.report_path:
        lines.append(f"Report: {result.report_path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the daily IV quality pipeline.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="process the latest stored snapshot instead of pulling new data")
    parser.add_argument("--send-alerts", action="store_true",
                        help=f"post alerts to Slack (needs {config.SLACK_WEBHOOK_ENV}); "
                             "default is a dry run that prints them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = get_connection()
    result = run_pipeline(con, skip_ingest=args.skip_ingest, send=args.send_alerts)
    print("\n" + summarize(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
