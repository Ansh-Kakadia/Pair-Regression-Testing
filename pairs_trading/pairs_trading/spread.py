"""
Hedge ratio estimation and spread construction for the KO/PEP pairs trading project.

Uses log prices throughout: regressing log(KO) on log(PEP) gives a β that is
scale-invariant and consistent with cointegration theory.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_hedge_ratio(
    prices: pd.DataFrame,
    dependent: str = "KO",
    independent: str = "PEP",
) -> tuple[float, float, object]:
    """
    OLS regression of log(dependent) on log(independent).

    Returns (alpha, beta, fitted_model).

    The choice of which series is dependent is asymmetric — OLS minimises
    vertical residuals, not perpendicular distance. For two assets it rarely
    changes the conclusion, but the β values will differ. We default to KO as
    dependent because KO is typically the smaller, more volatile series.
    """
    log_y = np.log(prices[dependent])
    log_x = np.log(prices[independent])

    X = sm.add_constant(log_x)
    X.columns = ["const", independent]
    model = sm.OLS(log_y, X).fit()

    alpha = model.params["const"]
    beta = model.params[independent]
    return alpha, beta, model


def build_spread(
    prices: pd.DataFrame,
    beta: float,
    dependent: str = "KO",
    independent: str = "PEP",
) -> pd.Series:
    """
    Construct the log-price spread: log(KO) - beta * log(PEP).

    This is the series we will test for stationarity and eventually trade.
    A stationary spread means the pair is cointegrated.
    """
    spread = np.log(prices[dependent]) - beta * np.log(prices[independent])
    spread.name = "spread"
    return spread


def zscore(series: pd.Series) -> pd.Series:
    """Normalise a series to zero mean, unit variance."""
    return (series - series.mean()) / series.std()


def estimate_half_life(spread: pd.Series) -> tuple[float, float, object]:
    """
    Estimate the mean-reversion half-life of a spread via AR(1) regression.

    Fits: Δspread_t = θ · spread_{t-1} + ε
    Half-life = ln(2) / θ  (days for daily data)

    Returns (half_life_days, theta, fitted_model).
    A negative or zero theta means the spread is not mean-reverting.
    """
    delta = spread.diff().dropna()
    lag   = spread.shift(1).dropna()

    aligned = pd.concat([delta, lag], axis=1).dropna()
    aligned.columns = ["delta", "lag"]

    model = sm.OLS(aligned["delta"], aligned["lag"]).fit()
    theta = -model.params["lag"]

    if theta <= 0:
        return float("inf"), theta, model

    half_life = np.log(2) / theta
    return half_life, theta, model


def rolling_half_life(spread: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute the half-life on a rolling basis to check stability over time.

    Returns a Series of half-life values (one per day) aligned to spread's index.
    """
    results = []
    for end in range(window, len(spread) + 1):
        window_spread = spread.iloc[end - window : end]
        hl, _, _ = estimate_half_life(window_spread)
        results.append((spread.index[end - 1], hl))

    idx, vals = zip(*results)
    return pd.Series(vals, index=idx, name="rolling_half_life")