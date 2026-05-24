"""
Step 2: Exploratory Visualization
Prices, returns, rolling correlation, and price ratio.
All plots saved to pairs_trading/plots/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pairs_trading.data import load_or_fetch
from pairs_trading.viz import (
    plot_prices,
    plot_returns,
    plot_rolling_correlation,
    plot_price_ratio,
)

TICKERS = ["KO", "PEP"]
START = "2018-01-01"
END = "2025-01-01"

if __name__ == "__main__":
    prices = load_or_fetch(TICKERS, START, END, name="ko_pep_7yr")

    print("Generating plots...")

    out = plot_prices(prices)
    print(f"  Plot 1 saved: {out}")

    out = plot_returns(prices)
    print(f"  Plot 2 saved: {out}")

    out = plot_rolling_correlation(prices, window=63)
    print(f"  Plot 3 saved: {out}")

    out = plot_price_ratio(prices)
    print(f"  Plot 4 saved: {out}")

    # Quick stats to print alongside the plots
    returns = prices.pct_change().dropna()
    print("\n--- Return stats ---")
    for ticker in TICKERS:
        r = returns[ticker] * 100
        print(f"{ticker}: mean={r.mean():.4f}%  std={r.std():.2f}%  "
              f"skew={r.skew():.2f}  kurt={r.kurtosis():.2f}")

    full_corr = returns["KO"].corr(returns["PEP"])
    rolling_corr = returns["KO"].rolling(63).corr(returns["PEP"])
    print(f"\nFull-sample return correlation : {full_corr:.3f}")
    print(f"Rolling 63-day corr (min/mean/max): "
          f"{rolling_corr.min():.3f} / {rolling_corr.mean():.3f} / {rolling_corr.max():.3f}")

    ratio = prices["KO"] / prices["PEP"]
    print(f"\nRaw KO/PEP price ratio: mean={ratio.mean():.4f}  std={ratio.std():.4f}")
    print("\nStep 2 complete.")
