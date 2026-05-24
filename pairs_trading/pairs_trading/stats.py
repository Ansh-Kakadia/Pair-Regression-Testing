"""
Cointegration and stationarity tests for the KO/PEP pairs trading project.

Test order:
  A. ADF on each price series individually  → confirm both are I(1)
  B. ADF on first differences               → confirm both become stationary
  C. Engle-Granger on the spread            → proper test when β is estimated
  D. Johansen on the price pair             → symmetric cross-check

Never use vanilla ADF on a regression residual when β was estimated from the
same data — the critical values are too lenient. Always use Engle-Granger (coint).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adf_test(series: pd.Series, label: str = "") -> dict:
    """
    Run ADF test on a single series. Returns a results dict.

    Null hypothesis: the series has a unit root (is non-stationary).
    Rejecting the null (p < 0.05) means the series is stationary.
    Failing to reject means we cannot conclude stationarity — NOT the same
    as proving a unit root exists.
    """
    stat, pval, lags, _, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {
        "label":    label,
        "stat":     stat,
        "pval":     pval,
        "lags":     lags,
        "crit_1pct": crit["1%"],
        "crit_5pct": crit["5%"],
        "crit_10pct": crit["10%"],
        "stationary_at_5pct": pval < 0.05,
    }


def print_adf(result: dict) -> None:
    sig = "STATIONARY" if result["stationary_at_5pct"] else "non-stationary (unit root not rejected)"
    print(f"  {result['label']:<30}  stat={result['stat']:7.3f}  p={result['pval']:.4f}  "
          f"5%crit={result['crit_5pct']:.3f}  lags={result['lags']}  → {sig}")


def engle_granger_test(
    prices: pd.DataFrame,
    dependent: str = "KO",
    independent: str = "PEP",
) -> dict:
    """
    Engle-Granger cointegration test using statsmodels coint().

    Uses stricter critical values than vanilla ADF because β is estimated
    from the same data (mining the data for the best-fitting β makes the
    residuals look more stationary than they really are).

    Null hypothesis: no cointegration (spread is a random walk).
    Rejecting (p < 0.05) means the pair is cointegrated.
    """
    y = np.log(prices[dependent])
    x = np.log(prices[independent])
    stat, pval, crit = coint(y, x, trend="c", autolag="AIC")
    return {
        "stat":      stat,
        "pval":      pval,
        "crit_1pct": crit[0],
        "crit_5pct": crit[1],
        "crit_10pct": crit[2],
        "cointegrated_at_5pct": pval < 0.05,
    }


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> object:
    """
    Johansen cointegration test.

    Symmetric — no arbitrary choice of dependent variable. Returns the raw
    statsmodels result object; callers should inspect:
      result.lr1    : trace statistics
      result.cvt    : critical values [90%, 95%, 99%]
      result.evec   : cointegrating vectors (columns)
    """
    log_prices = np.log(prices)
    return coint_johansen(log_prices, det_order, k_ar_diff)


def rolling_eg_pvalue(
    prices: pd.DataFrame,
    window: int = 252,
    dependent: str = "KO",
    independent: str = "PEP",
) -> pd.Series:
    """
    Compute the Engle-Granger p-value on a rolling window.

    Shows whether cointegration is stable over time or only present in
    certain regimes. A pair that cointegrates 80% of rolling windows is
    much more tradeable than one that barely passes on the full sample.
    """
    pvals = []
    idx   = []
    log_y = np.log(prices[dependent])
    log_x = np.log(prices[independent])

    for end in range(window, len(prices) + 1):
        y_win = log_y.iloc[end - window : end]
        x_win = log_x.iloc[end - window : end]
        _, pval, _ = coint(y_win, x_win, trend="c", autolag="AIC")
        pvals.append(pval)
        idx.append(prices.index[end - 1])

    return pd.Series(pvals, index=idx, name="eg_pvalue")