# iv-quality-monitor

A data quality monitor for derivatives implied-volatility data (equity and
commodity-proxy ETF options). Python + DuckDB.

**Status:** Stage 1 (ingestion into DuckDB). Work in progress.

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
