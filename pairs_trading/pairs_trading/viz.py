"""
Visualization for the KO/PEP pairs trading project.

Conventions:
- KO = Coke red, PEP = Pepsi blue — consistent across every plot
- Time always on x-axis with proper date formatting
- Save to file; never rely on interactive display from scripts
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

KO_COLOR = "#F40009"   # Coke red
PEP_COLOR = "#004B93"  # Pepsi blue

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def _format_dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.figure.autofmt_xdate(rotation=0, ha="center")


def plot_prices(prices: pd.DataFrame, save_as: str = "plot1_prices.png") -> Path:
    """Raw prices (left) and log prices (right) side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (title, transform) in zip(
        axes,
        [("Adjusted Close Prices ($)", lambda x: x),
         ("Log Prices (log $)", np.log)],
    ):
        ax.plot(prices.index, transform(prices["KO"]),
                color=KO_COLOR, label="KO", linewidth=1.4)
        ax.plot(prices.index, transform(prices["PEP"]),
                color=PEP_COLOR, label="PEP", linewidth=1.4)
        ax.set_title(title)
        ax.legend()
        _format_dates(ax)

    fig.suptitle("KO vs PEP — 2018–2024", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / save_as
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_returns(prices: pd.DataFrame, save_as: str = "plot2_returns.png") -> Path:
    """Daily returns time series (top) and histogram (bottom) for each ticker."""
    returns = prices.pct_change().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Time series
    for ax, (ticker, color) in zip(axes[0], [("KO", KO_COLOR), ("PEP", PEP_COLOR)]):
        ax.plot(returns.index, returns[ticker] * 100,
                color=color, linewidth=0.6, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ticker} daily returns (%)")
        _format_dates(ax)

    # Histogram
    for ax, (ticker, color) in zip(axes[1], [("KO", KO_COLOR), ("PEP", PEP_COLOR)]):
        r = returns[ticker] * 100
        ax.hist(r, bins=80, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(r.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={r.mean():.3f}%")
        ax.axvline(r.mean() + 2 * r.std(), color="gray", linestyle=":",
                   linewidth=1, label=f"±2σ ({2*r.std():.2f}%)")
        ax.axvline(r.mean() - 2 * r.std(), color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Daily return (%)")
        ax.set_title(f"{ticker} return distribution  |  σ={r.std():.2f}%  skew={r.skew():.2f}")
        ax.legend(fontsize=8)

    fig.suptitle("KO vs PEP — Daily Returns 2018–2024", fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / save_as
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_rolling_correlation(
    prices: pd.DataFrame,
    window: int = 63,
    save_as: str = "plot3_rolling_corr.png",
) -> Path:
    """Rolling 63-day correlation of daily returns."""
    returns = prices.pct_change().dropna()
    rolling_corr = returns["KO"].rolling(window).corr(returns["PEP"])
    mean_corr = rolling_corr.mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_corr.index, rolling_corr, color="black", linewidth=1.2,
            label="Rolling corr")
    ax.axhline(mean_corr, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_corr:.2f}")
    ax.axhline(0.3, color="red", linestyle=":", linewidth=1, alpha=0.6,
               label="Warning threshold (0.30)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.set_title(f"Rolling {window}-day Correlation of KO/PEP Daily Returns")
    ax.set_ylim(-0.2, 1.0)
    ax.legend()
    _format_dates(ax)

    plt.tight_layout()
    out = PLOTS_DIR / save_as
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def plot_price_ratio(prices: pd.DataFrame, save_as: str = "plot4_ratio.png") -> Path:
    """Raw KO/PEP price ratio over time — a quick preview of the spread structure."""
    ratio = prices["KO"] / prices["PEP"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ratio.index, ratio, color="purple", linewidth=1.2)
    ax.axhline(ratio.mean(), color="black", linestyle="--", linewidth=1,
               label=f"Mean ratio = {ratio.mean():.3f}")
    ax.fill_between(ratio.index,
                    ratio.mean() - ratio.std(),
                    ratio.mean() + ratio.std(),
                    color="purple", alpha=0.1, label="±1σ band")
    ax.set_title("KO / PEP Price Ratio (no hedge ratio yet)")
    ax.legend()
    _format_dates(ax)

    plt.tight_layout()
    out = PLOTS_DIR / save_as
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out
