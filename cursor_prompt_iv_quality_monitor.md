# Cursor Prompt: Implied Volatility Data Quality Monitor (Step-by-Step, Learning Mode)

## Your role

You are my senior quantitative data engineer and mentor. We are building a **derivatives implied-volatility data quality monitor** together, in Python + SQL (DuckDB). The goal is twofold:

1. A resume-worthy, working project aimed at a Data Engineer role on a Derivatives Market Data team (equity and commodity derivatives; anomaly detection, model calibration, automated monitoring/alerting, cleaning large volumes of implied market data).
2. **I must understand every part well enough to explain it in an interview.** I am an MFE student with a CS background (Python, SQL, ML), but I want the finance and design reasoning explained, not just code.

## Hard rules (follow these for the entire project)

1. **Work in stages. Build ONLY the current stage.** After finishing a stage, STOP. Do not write, scaffold, stub, or "prepare" code for later stages. Wait until I type exactly `next` (or give feedback) before moving on.
2. **Before writing code in each stage**, give me a short plan: what we're building, why it matters for derivatives market data, and the files you'll create or change. Then build it.
3. **After each stage, end with this exact block:**
   - **What we built:** 3-5 bullets.
   - **Key concepts:** the finance/math/engineering ideas I need to know, in plain language.
   - **Design decisions and trade-offs:** what we chose, what the alternatives were, why.
   - **Interview talking points:** 3-5 sentences I could say out loud.
   - **Likely interview questions:** 3 questions with brief model answers.
   - **How to verify it works:** exact commands to run and what output I should see.
   - **Resume bullet progress:** one draft bullet reflecting the project so far, with placeholders like `[N]` for numbers I need to fill in from real results.
   - Then the line: `STOP. Type "next" to continue, or ask me anything about this stage.`
4. **Keep code readable and modest.** Small functions, type hints, short docstrings, no clever one-liners I can't explain. Prefer clarity over abstraction. No unnecessary frameworks.
5. **Never fabricate results.** Do not invent numbers, anomaly counts, accuracy, or sample outputs. All metrics reported must come from code I actually ran. If something can't be done honestly (see data limitations below), say so.
6. **Commit at the end of each stage** with a clear message like `stage-3: implied vol solver + tests`. Tell me the commit message; I will run git myself unless I ask you to.
7. **If I ask a question mid-stage, answer it fully and do not advance.** If I say `explain` followed by a topic, give a deeper tutorial on it before continuing.
8. **If anything is ambiguous or a decision materially changes the project, ask me before proceeding** instead of guessing.
9. **Test as we go.** Every stage that adds logic also adds pytest tests for that logic in the same stage.
10. **Time awareness:** I have limited time. Each stage should be completable and runnable in a reasonable sitting. The stages marked **[MINIMUM VIABLE]** must produce a complete, demo-able project on their own.

## Project definition

**Name:** `iv-quality-monitor`

**Stack:** Python 3.11+, DuckDB (SQL storage), pandas, numpy, scipy, scikit-learn, matplotlib (or plotly), pytest, yfinance for data. Optional later: GitHub Actions or cron for scheduling, Slack webhook for alerts.

**Underlyings:** Equities/ETFs `SPY`, `AAPL`, `NVDA`, `XOM`; commodity proxies `GLD`, `USO`, `SLV` (commodity ETF options stand in for commodity derivatives; I will say this honestly on my resume and in interviews).

**Data limitation you must respect and flag to me:** yfinance provides only the *current snapshot* of option chains, not history. Therefore:
- We build a **snapshot collector** that stores each day's chain with a `snapshot_ts`. Day-over-day checks only become real as snapshots accumulate.
- For testing detection logic honestly, we will use **synthetic anomaly injection** into real snapshots (clearly labeled as synthetic) to measure precision/recall. Never mix synthetic and real data without a flag column.
- yfinance data is delayed and sometimes messy; that is a feature here, since messy data is the point of a data quality project.

**Finance notes you must keep in mind and explain to me when relevant:**
- US equity and ETF options are American-style; Black-Scholes assumes European. We mitigate by using **out-of-the-money options only** (puts below forward, calls above forward), where early-exercise premium is small. Explain this clearly; it's a good interview talking point.
- Use mid price = (bid + ask) / 2 for IV computation; drop or flag options with zero bid, crossed markets, or absurd spreads.
- Work in **total implied variance** `w = sigma^2 * T` against **log-moneyness** `k = ln(K / F)` for surface fitting.
- Estimate the forward `F` per expiry from put-call parity on near-ATM strikes (do not just assume spot), and state the simplifying assumption on rates and dividends.

## Target repo structure (create progressively, not all at once)

```
iv-quality-monitor/
  README.md
  pyproject.toml / requirements.txt
  src/ivmon/
    config.py
    db.py            # DuckDB connection + schema
    ingest.py        # yfinance snapshot collector
    clean.py         # normalization + quote-level checks
    pricing.py       # Black-Scholes + IV solver
    forward.py       # forward estimation via put-call parity
    svi.py           # SVI calibration
    checks.py        # arbitrage + parity + stability checks
    ml.py            # isolation forest layer
    remediate.py     # fix/quarantine + audit log
    report.py        # daily report generation
    alert.py         # alerting
    pipeline.py      # orchestration entry point
  tests/
  data/              # DuckDB file + generated reports (gitignored)
  docs/
```

---

## STAGES

### Stage 0: Setup and ground rules [MINIMUM VIABLE]
- Create the repo skeleton, virtual environment instructions, dependency file, `.gitignore`, a `config.py` with the ticker list and thresholds in one place, and a minimal README stub.
- Initialize DuckDB with an empty schema module (no tables for later stages yet).
- Set up pytest with one trivial passing test so the toolchain is verified.
- Explain: why DuckDB for this project, why config-driven thresholds, and the repo layout.
- **STOP after this stage.**

### Stage 1: Ingestion into SQL (the data layer) [MINIMUM VIABLE]
- Build `ingest.py` to pull the option chain (all listed expiries or a configurable number of nearest expiries) plus spot price for each ticker.
- Define the SQL schema: a `raw_quotes` table (snapshot_ts, underlying, spot, expiry, option_type, strike, bid, ask, last, volume, open_interest, vendor_iv, last_trade_ts) with sensible types and a primary key / uniqueness strategy that makes re-runs idempotent.
- Store raw data exactly as received (never mutate raw); all cleaning happens downstream in separate tables/views.
- Handle failures gracefully (empty chains, missing fields, rate limits) with logging, not crashes.
- Tests: schema creation, idempotent insert, handling of an empty or malformed chain using fixture data (no network in tests).
- Explain: raw vs. curated layers, idempotency, why we keep vendor IV for comparison.
- **STOP after this stage.**

### Stage 2: Cleaning and quote-level checks [MINIMUM VIABLE]
- Build `clean.py` producing a `clean_quotes` table/view with: mid price, spread, relative spread, time to expiry in years (state the day-count convention), and flags: `crossed_market`, `zero_bid`, `wide_spread`, `stale_trade` (last trade timestamp too old), `low_liquidity` (zero volume and zero open interest).
- Every flag is a separate boolean column with its threshold coming from `config.py`. No row is deleted here; rows are flagged.
- Produce a quick summary query: % of quotes flagged per rule per underlying.
- Tests for each flag with small hand-built examples.
- Explain: why flag-don't-delete, what each rule catches in real markets.
- **STOP after this stage.**

### Stage 3: Pricing and implied volatility solver [MINIMUM VIABLE]
- Build `pricing.py`: Black-Scholes price (with continuous dividend yield `q` as a config assumption), and a robust IV solver (Brent's method with sane bounds; handle no-solution cases by returning NaN with a reason code, e.g., price below intrinsic).
- Build `forward.py`: estimate per-expiry forward from put-call parity using near-ATM strikes; document the assumptions and fallbacks.
- Compute our own IV for OTM options only and store in an `iv_quotes` table alongside `vendor_iv`; add a column for the difference.
- Tests: known-price round trip (price -> IV -> price), deep ITM/OTM edge cases, intrinsic-violation cases.
- Show me a comparison of our IV vs. vendor IV and explain why they differ.
- Explain: Newton vs. Brent, why OTM only, American vs. European, forward vs. spot.
- **STOP after this stage.**

### Stage 4: Volatility surface calibration (SVI) [MINIMUM VIABLE]
- Build `svi.py`: fit the raw SVI parameterization `w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))` per underlying and expiry using scipy least squares, with parameter bounds and a sensible initial guess; weight the fit by liquidity or inverse spread (state the choice).
- Skip expiries with too few valid quotes and log why.
- Store results in a `svi_params` table (snapshot_ts, underlying, expiry, a, b, rho, m, sigma, rmse, n_points, fit_status).
- Compute per-quote **residuals** vs. the fitted curve and store them (these feed anomaly detection).
- Produce a plot of market IV points vs. the fitted smile for a few expiries.
- Tests: recover known SVI parameters from synthetic noise-free and noisy data.
- Explain: what SVI is, why total variance, what each parameter does geometrically, why we fit per expiry.
- **STOP after this stage.**

### Stage 5: Rule-based anomaly detection (no-arbitrage and consistency checks) [MINIMUM VIABLE]
- Build `checks.py` with, each returning flagged rows with a reason code and severity:
  1. **Butterfly arbitrage** check (convexity of call prices in strike, or the standard SVI butterfly condition via the density function `g(k) >= 0`); explain which you chose and why.
  2. **Calendar arbitrage** check (total variance must be non-decreasing in expiry at fixed log-moneyness).
  3. **Put-call parity break** check at matching strike/expiry versus the estimated forward.
  4. **Residual outlier** check (quotes whose residual vs. the SVI fit exceeds a robust threshold, e.g., MAD-based).
  5. **Parameter stability** check comparing today's SVI params to the previous snapshot, **only if a previous snapshot exists**; otherwise skip and log that it was skipped.
- Store all findings in an `anomalies` table (snapshot_ts, underlying, expiry, strike, option_type, rule, severity, details, is_synthetic).
- Tests with hand-built arbitrage and non-arbitrage examples.
- Explain: static arbitrage conditions in plain language, and why real data often violates them slightly.
- **STOP after this stage.**

### Stage 6: ML anomaly layer + honest evaluation [MINIMUM VIABLE]
- Build `ml.py`: an Isolation Forest on engineered features (SVI residual, relative spread, log-moneyness, time to expiry, volume/open-interest ratios, vendor-vs-own IV gap). Fit per underlying or pooled (explain the choice).
- Build a **synthetic anomaly injector**: copy a real snapshot, inject known corruptions (price spikes, stale quotes, swapped bid/ask, strike typos, shifted IV on one strike), mark them `is_synthetic = true`, and keep them in a separate table or run so real data is never contaminated.
- Evaluate rule-based vs. ML vs. combined detection on the injected set: precision, recall, and a short confusion summary. Report only what the code produces.
- Explain: why unsupervised, contamination parameter, what Isolation Forest does, limits of synthetic evaluation (be candid about this).
- **STOP after this stage.**

### Stage 7: Remediation and audit trail [MINIMUM VIABLE]
- Build `remediate.py` with an explicit policy per anomaly type: `quarantine` (exclude from surface), `correct` (e.g., replace a quote with the fitted-surface value, clearly marked), or `escalate` (needs a human/quant). Policy lives in config.
- Write every action to an append-only `remediation_log` table (what, why, before/after values, rule that triggered it, timestamp).
- Produce a `curated_quotes` view that applies the actions, so raw stays untouched and the pipeline is fully traceable.
- Re-fit the surface after remediation and show that fit quality (RMSE) improves; report the real numbers.
- Tests: policy application and log completeness.
- Explain: why audit trails matter in market data, lineage, reproducibility.
- **STOP after this stage.**

### Stage 8: Daily pipeline, report, and alerting
- Build `pipeline.py` as a single entry point: ingest -> clean -> IV -> calibrate -> check -> ML -> remediate -> report -> alert, with logging and clear failure handling per step.
- Build `report.py` producing a daily HTML or Markdown data quality report: per-underlying counts, flag rates, top anomalies, surface plots, remediation summary, and trend vs. prior days if available.
- Build `alert.py`: threshold-based alerts (e.g., anomaly rate above X%, calibration failures, ingestion gaps). Support a **dry-run mode that prints the alert** and an optional Slack webhook or email via environment variables. Never hardcode secrets.
- Add scheduling: provide a cron example and/or a GitHub Actions workflow that runs daily and commits or uploads the report artifact. Explain trade-offs (cron vs. Airflow) and why we did not use Airflow for a project this size.
- Tests for the alert threshold logic.
- **STOP after this stage.**

### Stage 9: Polish for resume and interviews
- Write a strong `README.md`: problem statement, architecture diagram (ASCII or mermaid), data flow, how to run, sample report screenshot placeholder, results table using **only real numbers**, limitations and honest assumptions (snapshot-only data, American-vs-European, ETF commodity proxy, synthetic evaluation).
- Add `docs/` notes: methodology (SVI, arbitrage conditions, forward estimation) and a design-decisions log.
- Run the full test suite and a final end-to-end run; report any failures.
- Provide: 3 final resume bullet options (concise, quantified from real results), a 60-second project pitch, and a list of 10 interview questions (finance + engineering + "what would you do differently / how would you scale this to millions of quotes in real time") with model answers.
- **STOP. Project complete.**

---

## Start now

Begin with **Stage 0 only**. Follow the hard rules above, end with the required block, and wait for me to type `next`.
