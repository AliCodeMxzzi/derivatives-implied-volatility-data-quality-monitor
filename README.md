# iv-quality-monitor

A data quality monitor for derivatives implied-volatility data (equity and
commodity-proxy ETF options). Python + DuckDB.

**Status:** Stage 8 complete (end-to-end pipeline, report, alerts, scheduling). Final polish pending.

Run everything with one command (pull during US market hours):

```powershell
python -m ivmon.pipeline                # ingest -> ... -> report -> alerts (dry run)
python -m ivmon.pipeline --skip-ingest  # re-process the latest stored snapshot
python -m ivmon.pipeline --send-alerts  # post to Slack if SLACK_WEBHOOK_URL is set
```

Reports land in `data/reports/`. Scheduling options: see `docs/scheduling.md`.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest
```

## Collect a snapshot

```powershell
python -m ivmon.ingest
```

Pulls the nearest expiries for each ticker in `config.py` into
`data/ivmon.duckdb` (table `raw_quotes`). yfinance only serves the current
chain, so run this regularly to build history.

## Layout

- `src/ivmon/` - package code
- `tests/` - pytest tests
- `data/` - DuckDB file and generated reports (gitignored)
- `docs/` - methodology notes (added later)
