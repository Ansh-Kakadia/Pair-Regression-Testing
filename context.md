# KO/PEP Pairs Trading — Session Context

**Date:** 2026-05-24  
**Project:** Multi-session pairs trading analysis on Coca-Cola (KO) and PepsiCo (PEP)  
**Goal:** Build a complete 9-step pipeline with real statistical results and deep understanding of the methodology

---

## What we built today

### Environment setup
- Created `.venv` virtual environment in project root
- Installed: `yfinance`, `pandas`, `pyarrow`, `statsmodels`, `scipy`, `matplotlib`, `lxml`, `jupyterlab`, `ipykernel`
- Registered venv as Jupyter kernel: `pairs-trading`
- Kernel launch command: `.venv\Scripts\jupyter.exe lab --notebook-dir=pairs_trading`

### Project structure
```
Pair-Regression-Testing/
  .venv/                         # virtual environment
  requirements.txt               # yfinance, pandas, pyarrow, statsmodels, scipy, matplotlib, lxml
  context.md                     # this file
  pairs_trading/
    data/
      ko_pep_7yr.parquet         # cached price data (2018-01-01 to 2025-01-01)
    notebooks/
      step1_data.ipynb
      step2_viz.ipynb
      step3_hedge_ratio.ipynb
      step4_halflife.ipynb
    pairs_trading/               # Python module
      __init__.py
      data.py                    # fetch_prices, validate_prices, load_or_fetch, fetch_earnings_dates
      spread.py                  # estimate_hedge_ratio, build_spread, zscore, estimate_half_life, rolling_half_life
      viz.py                     # plot_prices, plot_returns, plot_rolling_correlation, plot_price_ratio
    plots/                       # saved PNGs from step2 script
    step1_fetch.py               # standalone runner for step 1
    step2_viz.py                 # standalone runner for step 2
    tests/
      __init__.py
```

---

## Steps completed

### Step 1 — Data Acquisition (`step1_data.ipynb`)
Pulled 7 years of adjusted close prices via yfinance, validated, and cached to Parquet.

**Key results:**
- 1,761 trading days (2018-01-02 to 2024-12-31) — exactly as expected
- No NaN values, no bad prices, no long date gaps — clean data
- KO range: $31–$69 | PEP range: $75–$177 (both adjusted for dividends and splits)
- Earnings dates fetched for both tickers (noted: large PEP EPS misses in mid-2025)

---

### Step 2 — Exploratory Visualization (`step2_viz.ipynb`)
Prices, log prices, daily returns, rolling correlation, and raw price ratio. All plots inline with full axis labels and plain-English explanations.

**Key results:**
- Full-sample return correlation: **0.737**
- Rolling 63-day correlation: min **0.258** / mean **0.703** / max **0.946**
- KO daily volatility (σ): **1.23%** | PEP: **1.29%** — nearly identical, good for pairing
- PEP kurtosis: **~20** — a handful of extreme days (likely large earnings misses) create very fat tails
- Raw KO/PEP price ratio: mean **0.375**, CV **0.073** — reasonably stable

---

### Step 3 — OLS Hedge Ratio & Spread Construction (`step3_hedge_ratio.ipynb`)
Regressed log(KO) on log(PEP) to find β, then constructed the spread: `log(KO) − β × log(PEP)`.

**Key results:**
- **β = 0.797** — for every 1% PEP moves, KO moves ~0.80%
- **R² = 0.91** — 91% of KO's log-price variation is explained by PEP alone
- α ≈ −0.002 (essentially zero — clean relationship)
- Asymmetry check: reversing regression gives implied β = 0.876 — stable range, conclusion unchanged
- Spread: 5.5% of days outside ±2σ (expected ~5%), 0 days outside ±4σ, 77 zero-crossings
- **The spread looks mean-reverting visually** — formal test is Step 5

---

### Step 4 — Mean-Reversion Speed & Half-Life (`step4_halflife.ipynb`)
Fit an AR(1) regression on the spread to estimate the Ornstein-Uhlenbeck speed parameter θ and half-life.

**Key results:**
- **θ = 0.00963**
- **Full-sample half-life: 72 days (~3.4 months)**
- AR(1) p-value: **0.0063** — mean reversion is statistically real
- Rolling 252-day half-life: min **15.7 days** / mean **∞** (some windows had θ ≤ 0)
- Only **10.1%** of rolling windows fall in the 5–30 day "sweet spot"

**What this means:** 72 days is outside the ideal 5–30 day range. Capital is tied up ~6 months per full round-trip. This is normal for slow-moving consumer staples. Some rolling windows showed mean reversion breaking down entirely (half-life → ∞) — a risk the backtest must grapple with. The rolling minimum of 15.7 days shows the pair *can* revert quickly in some regimes.

---

## Key decisions & conventions established
- **Log prices** used throughout (not raw prices) — theoretically correct for cointegration
- **KO as dependent variable** in OLS — choice is flagged as asymmetric, both directions checked
- **KO = red (#F40009), PEP = blue (#004B93)** — consistent across all plots
- **All axis labels must be in plain English** with units and context
- **All markdown must use actual Unicode characters** (θ, μ, σ) — not `θ` escape sequences
- **Data never re-downloaded** — always load from `ko_pep_7yr.parquet` cache

---

## Next steps (Steps 5–9)

### Step 5 — Cointegration Testing *(most important remaining step)*
This is the formal statistical proof. All three tests:
1. **ADF** on each series individually — confirm both are I(1) non-stationary
2. **Engle-Granger** — proper test on spread residuals (uses stricter critical values than vanilla ADF because β was estimated from the data)
3. **Johansen** — symmetric test, no arbitrary dependent variable choice

Expected: Engle-Granger should reject the null at 5% with 1,761 days. If it doesn't, investigate before moving forward.

Module to write: `pairs_trading/stats.py`

---

### Step 6 — (merged into Step 4)
OU fit and half-life are already done in Step 4. Step 6 is complete.

---

### Step 7 — Trading Rule Design
- Define entry/exit z-score thresholds (typical: entry ±2, exit 0, stop ±4)
- Discuss how the 72-day half-life affects threshold choice
- Keep thresholds simple — no grid-search overfitting on full sample

---

### Step 8 — Walk-Forward Backtest
- Train window: 252 days | Test window: 63 days (rolling)
- Fit β, spread mean/std on training data only — no peeking
- Cost model: 10 bps/round-trip + ~40 bps/year short borrow
- Report: Sharpe, max drawdown, win rate, # trades
- Honest expectation: Sharpe 0.3–1.0. Anything above 1.5 is probably a bug.

Module to write: `pairs_trading/backtest.py`

---

### Step 9 — Sanity Checks
- Compare results to a random pair (null distribution)
- Bootstrap Sharpe confidence intervals — does the CI include zero?
- Threshold sensitivity: does the backtest hold up if entry z changes from 2.0 to 1.5 or 2.5?

Module to write: `pairs_trading/sanity.py`