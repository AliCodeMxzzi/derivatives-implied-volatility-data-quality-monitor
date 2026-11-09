# Scheduling the daily run

The entry point is one command:

```
python -m ivmon.pipeline                 # pull a new snapshot, process it, report, alert (dry run)
python -m ivmon.pipeline --skip-ingest   # re-process the latest stored snapshot
python -m ivmon.pipeline --send-alerts   # also post alerts to Slack (needs SLACK_WEBHOOK_URL)
```

Exit code is 0 when every step succeeded and 1 if any step failed, so any
scheduler can treat a failed run as a failure.

Pull during US market hours (about 6:30am to 1:00pm Pacific). Outside them
Yahoo returns 0/0 quotes and the `zero_bid_rate` alert fires.

## Option 1: cron on an always-on Linux machine (durable history)

```
# 11:00 Pacific on weekdays (machine timezone = America/Los_Angeles)
0 11 * * 1-5  cd /path/to/repo && .venv/bin/python -m ivmon.pipeline --send-alerts >> data/pipeline.log 2>&1
```

## Option 2: Windows Task Scheduler (this project's dev machine)

```
schtasks /Create /TN "IVMonitor" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 11:00 ^
  /TR "\"C:\path\to\repo\.venv\Scripts\python.exe\" -m ivmon.pipeline" ^
  /SD 01/01/2027
```
(Run it with the repo as the working directory, e.g. by wrapping it in a .bat file.)
The machine must be on at that time.

## Option 3: GitHub Actions (`.github/workflows/daily.yml`)

Runs on a schedule, uploads the report as an artifact, and posts to Slack if the
`SLACK_WEBHOOK_URL` repository secret exists. The workflow has not been run from
this machine, so treat the first run as a test (use "Run workflow" manually).

Limitation: runners are wiped after every run, so the workflow keeps the DuckDB
file in the Actions cache. Caches can be evicted, so history is best-effort.
GitHub scheduled runs may also start several minutes late.

## Cron vs Airflow, and why not Airflow here

| | cron / Task Scheduler / Actions | Airflow |
|---|---|---|
| Setup | one line / one file | scheduler, web server, metadata DB, workers |
| Retries, backfills, dependency graph, UI | none built in (we handle step failure inside the pipeline) | built in |
| Fits when | one job, one machine, a daily cadence | dozens of interdependent jobs, several teams, backfills |

This project is one daily job with eight sequential steps, and the step
dependency handling already lives in `pipeline.py`. Airflow would add more
operational surface (a service to run, monitor and upgrade) than the pipeline
itself. If the number of jobs or data sources grew, or we needed backfills with
per-task retries and a UI, that is the point to move to Airflow (or Dagster/Prefect).
