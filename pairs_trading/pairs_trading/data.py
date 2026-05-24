"""
Data acquisition and caching for the KO/PEP pairs trading project.

Pull once → validate → save to Parquet → all downstream work reads from cache.
Never re-download inside an analysis loop.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Pull adjusted close prices for a list of tickers from Yahoo Finance.

    Returns a DataFrame with dates as index and tickers as columns.
    auto_adjust=True handles both splits AND dividends — essential for KO/PEP
    which pay quarterly dividends that would otherwise look like real price drops.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if len(tickers) > 1:
        df = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})

    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def validate_prices(df: pd.DataFrame, tickers: list[str]) -> dict:
    """
    Run sanity checks on raw price data. Returns a dict of findings.
    An empty dict means no issues found. Never silently passes problems.
    """
    findings = {}

    missing = set(tickers) - set(df.columns)
    if missing:
        findings["missing_tickers"] = missing

    nan_counts = df.isna().sum()
    if nan_counts.any():
        findings["nan_counts"] = nan_counts[nan_counts > 0].to_dict()

    # Gaps > 7 calendar days suggest something wrong (not just weekends/holidays)
    date_gaps = df.index.to_series().diff()
    long_gaps = date_gaps[date_gaps > pd.Timedelta(days=7)]
    if len(long_gaps) > 0:
        findings["long_date_gaps"] = {str(k): str(v) for k, v in long_gaps.items()}

    bad_prices = (df <= 0).any()
    if bad_prices.any():
        findings["zero_or_negative_prices"] = bad_prices[bad_prices].index.tolist()

    # Flag extreme single-day moves (>30%) — possible but worth investigating
    returns = df.pct_change()
    extreme = (returns.abs() > 0.30).sum()
    if extreme.any():
        findings["extreme_daily_moves"] = extreme[extreme > 0].to_dict()

    return findings


def load_or_fetch(
    tickers: list[str],
    start: str,
    end: str,
    name: str,
) -> pd.DataFrame:
    """
    Load prices from Parquet cache if available, otherwise fetch from Yahoo Finance.

    Always prints validation findings so problems surface immediately.
    """
    cache_path = DATA_DIR / f"{name}.parquet"

    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Fetching {tickers} from {start} to {end} ...")
    df = fetch_prices(tickers, start, end)

    findings = validate_prices(df, tickers)
    if findings:
        print("WARNING — data quality findings:")
        for k, v in findings.items():
            print(f"  {k}: {v}")
    else:
        print("Validation passed — no issues found.")

    df.to_parquet(cache_path)
    print(f"Cached to {cache_path}")
    return df


def fetch_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Pull recent + projected earnings dates for a ticker.
    Cache these alongside prices for plot annotations and backtest exclusion rules.
    """
    return yf.Ticker(ticker).get_earnings_dates(limit=40)


def describe_prices(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded price data."""
    print(f"\n--- Price data summary ---")
    print(f"Date range : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Trading days: {len(df)}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nLatest prices:")
    print(df.tail(3).to_string())
    print(f"\nBasic stats:")
    print(df.describe().to_string())
