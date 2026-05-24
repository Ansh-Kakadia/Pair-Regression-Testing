# Pairs Trading Pipeline — Session Context

**Last updated:** 2026-05-24  
**Project:** Multi-session quantitative finance learning project — build a complete pairs trading pipeline, understand every step, and find a genuinely cointegrated pair.

---

## Environment

- `.venv` virtual environment in project root
- Installed: `yfinance`, `pandas`, `pyarrow`, `statsmodels`, `scipy`, `matplotlib`, `lxml`, `jupyterlab`, `ipykernel`
- Jupyter kernel name: `pairs-trading`
- Launch JupyterLab: `.venv\Scripts\jupyter.exe lab --notebook-dir=pairs_trading`

---

## Project structure

```
Pair-Regression-Testing/
  .venv/
  requirements.txt
  context.md                        ← this file
  pairs_trading/
    data/
      ko_pep_7yr.parquet            ← cached KO/PEP prices (2018-01-01 to 2025-01-01)
    notebooks/
      step1_data.ipynb
      step2_viz.ipynb
      step3_hedge_ratio.ipynb
      step4_halflife.ipynb
      step5_cointegration.ipynb
    pairs_trading/                  ← Python module (reusable for any pair)
      __init__.py
      data.py       fetch_prices, validate_prices, load_or_fetch, fetch_earnings_dates
      spread.py     estimate_hedge_ratio, build_spread, zscore, estimate_half_life, rolling_half_life
      viz.py        plot_prices, plot_returns, plot_rolling_correlation, plot_price_ratio
      stats.py      adf_test, engle_granger_test, johansen_test, rolling_eg_pvalue
    plots/
    step1_fetch.py
    step2_viz.py
    tests/
      __init__.py
```

---

## KO/PEP analysis — COMPLETE (pair rejected)

Ran all five steps on KO (Coca-Cola) vs PEP (PepsiCo), 2018-01-01 to 2025-01-01, 1,761 trading days.

### Step 1 — Data
- Clean data, no NaNs, no bad prices
- KO: $31–$69 | PEP: $75–$177 (adjusted)

### Step 2 — Exploratory viz
- Full-sample return correlation: 0.737
- Rolling 63-day correlation: min 0.258 / mean 0.703 / max 0.946
- Volatility nearly identical: KO σ = 1.23%, PEP σ = 1.29%
- PEP kurtosis ~20 — fat tails from large earnings misses

### Step 3 — Hedge ratio
- β = **0.797** (log-price OLS, KO as dependent)
- R² = **0.91**
- Spread: 5.5% of days outside ±2σ, 0 outside ±4σ, 77 zero-crossings
- Spread *looks* mean-reverting visually

### Step 4 — Half-life
- θ = 0.00963, half-life = **72 days** (~3.4 months)
- Well outside the 5–30 day sweet spot for daily trading
- AR(1) p-value = 0.0063 (statistically detectable but slow)
- Rolling min: 15.7 days — only 10.1% of windows in sweet spot
- Some rolling windows: half-life → ∞ (mean reversion broke down)

### Step 5 — Cointegration testing ← **PAIR REJECTED HERE**

| Test | Result |
|---|---|
| ADF on price levels | Both non-stationary ✓ (I(1) confirmed) |
| ADF on returns | Both stationary ✓ |
| Engle-Granger | p = 0.31 — **fail to reject** (no cointegration) |
| Johansen trace | 10.4 vs 95% CV 15.5 — **fail to reject** |
| Rolling EG (252d) | Only **2.3%** of windows significant (35 of 1,510) |

**Rolling cluster finding:** The 35 significant windows are not scattered — they cluster tightly between **2020-02-03 and 2022-01-24** (COVID crash through post-COVID reopening). This was a period of extreme macro stress where a common external shock temporarily made the pair look cointegrated. Outside this window: essentially zero evidence.

**Why KO/PEP failed:** PEP has structurally diverged from KO. PEP generates ~55% of revenue from snacks (Frito-Lay), making it a diversified food-and-beverage company, while KO remains a pure beverage company. The academic literature that established this as a canonical pair was written when both were symmetric beverage competitors. That structural similarity no longer holds.

**Note:** Vanilla ADF on the spread gave p = 0.13 (looked marginal) while Engle-Granger gave p = 0.31 (correctly stricter because β was estimated from the data). This is exactly the bias the project warned about — always use Engle-Granger, not vanilla ADF, when β is estimated.

---

## What to look for in the next pair

A good candidate needs:
1. **Structurally similar businesses** — same industry, same revenue model, similar macro exposure. The KO/PEP failure was a structural divergence lesson.
2. **Engle-Granger p < 0.05** on a 5–7 year window
3. **Half-life 5–30 days** — fast enough to trade, slow enough to execute
4. **Rolling cointegration stable** — significant in >50% of rolling windows, not just one clustered regime
5. **Return correlation > 0.70** — necessary (not sufficient) condition

**Candidate pairs to screen** (from original project ranking — revisit these):
- CVX / XOM — both pure-play integrated oil majors, very similar revenue structure
- UNP / CSX — Class I railroads, nearly identical business model
- WM / RSG — waste management duopoly, regulated, slow-moving
- DUK / SO — regulated electric utilities, similar rate-base structures
- LOW / HD — home improvement retail (both exposed to housing cycle)
- V / MA — payment network duopoly (but both have diverged into different geographies/products)

---

## Conventions established (carry forward to next pair)

- **Log prices** throughout — theoretically correct for cointegration
- **Engle-Granger, not vanilla ADF** — when β is estimated from data
- **Both EG and Johansen** must agree before accepting a pair
- **Rolling EG p-value plot** — highlight p < 0.05 windows in green against muted line; check clustered vs scattered
- **Color scheme**: first ticker = red, second = blue (pick new colors per pair)
- **All axis labels in plain English** with units
- **All markdown uses actual Unicode** (θ, μ, σ, —) — not escape sequences
- **Data cached to Parquet** — never re-download mid-analysis
- **Honest threshold**: Sharpe > 1.5 in backtest = suspect a bug

---

## Next session

1. User selects a new candidate pair
2. Reuse the existing pipeline — swap tickers in `load_or_fetch`, re-run steps 1–5
3. If Engle-Granger passes and half-life is in range, proceed to Steps 7–9 (trading rules, backtest, sanity checks)
4. Modules for Steps 7–9 still need to be written: `backtest.py`, `sanity.py`