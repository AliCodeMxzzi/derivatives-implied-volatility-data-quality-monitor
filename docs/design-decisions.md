# Design decisions log

| # | Decision | Why | Trade-off / what would change it |
|---|---|---|---|
| 1 | DuckDB, single file | Embedded columnar SQL, no server, fast analytics | Not for concurrent writers; use Postgres/ClickHouse at scale |
| 2 | Raw layer never mutated | Reproducibility and lineage | More storage; derived tables are rebuilt |
| 3 | Idempotent ingest (composite key, `INSERT OR IGNORE`) | Safe to re-run a snapshot | Silent skip of duplicates |
| 4 | Snapshot-based history | yfinance has only the current chain | Only as much history as the days we ran |
| 5 | Flag, don't delete | Keeps evidence, lets rules disagree | Downstream must filter |
| 6 | Black-76 on forward, OTM only | Limits American early-exercise error | Ignores that error rather than modelling it (would need a binomial/PDE model) |
| 7 | Brent solver with reason codes | Always converges; failures are explainable | Slower than Newton, irrelevant at this size |
| 8 | Parity forward, spot fallback | Market-implied, includes carry | Noisy with bad quotes; median of 5 strikes mitigates |
| 9 | SVI per expiry, soft_l1, spread weights | Robust fit, standard arbitrage tooling | Fits can hide real data errors; checks run on the fit |
| 10 | Stability compared as curves | Raw SVI params are not identifiable alone | More compute |
| 11 | Pooled Isolation Forest, 5% contamination | Small per-name samples | Result is near chance; kept as an evaluated layer |
| 12 | Synthetic injection in an isolated DB | Known ground truth without contaminating real data | Only measures my own corruption types |
| 13 | Policy-based remediation, default escalate | Humans decide the unknown | Quarantine/correct policy for residual outliers is unvalidated (refit got worse) |
| 14 | Append-only log plus run table | Auditability, no stale "latest" actions | Log grows; needs retention policy |
| 15 | Step dependency graph, report/alert always run | One failure should not hide the rest | More code than a linear script |
| 16 | Strictly-greater alert thresholds, dry run default | Predictable, safe to test | Thresholds are uncalibrated round numbers |
| 17 | Slack webhook only via env var | No secrets in the repo | Requires environment setup |
| 18 | cron/Task Scheduler/Actions, not Airflow | One daily job, one machine | Move to Airflow/Dagster for many jobs, backfills, per-task retries |

## What I would do differently
- Replace yfinance with a vendor feed that provides history, exchange timestamps and sizes.
- Add an American-exercise model (binomial) and real dividends and rates.
- Calibrate alert thresholds on weeks of history.
- Validate or drop the residual-outlier quarantine/correct policy, since it made the refit worse.
- Give ML a target: labelled vendor errors, or at least rank ML findings below rule-based ones in reports.
