# iv-quality-monitor

A data quality monitor for derivatives implied-volatility data (equity and
commodity-proxy ETF options). Python + DuckDB.

**Status:** Stage 0 (setup). Work in progress.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Layout

- `src/ivmon/` - package code
- `tests/` - pytest tests
- `data/` - DuckDB file and generated reports (gitignored)
- `docs/` - methodology notes (added later)
