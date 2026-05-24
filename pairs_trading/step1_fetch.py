"""
Step 1: Data Acquisition
Pull 7 years of adjusted closes for KO and PEP, validate, and cache to Parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pairs_trading.data import load_or_fetch, fetch_earnings_dates, describe_prices

TICKERS = ["KO", "PEP"]
START = "2018-01-01"
END = "2025-01-01"

if __name__ == "__main__":
    prices = load_or_fetch(TICKERS, START, END, name="ko_pep_7yr")
    describe_prices(prices)

    print("\n--- Fetching earnings dates ---")
    for ticker in TICKERS:
        try:
            earnings = fetch_earnings_dates(ticker)
            print(f"\n{ticker} earnings (most recent 5):")
            print(earnings.head(5).to_string())
        except Exception as e:
            print(f"  {ticker} earnings fetch failed: {e}")

    print("\nStep 1 complete. Data cached to pairs_trading/data/ko_pep_7yr.parquet")
