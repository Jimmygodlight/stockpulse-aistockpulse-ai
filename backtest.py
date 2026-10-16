import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]

PRICES_TABLE = (
    f"{PROJECT_ID}.stockpulse.prices"
)

RESULTS_TABLE = (
    f"{PROJECT_ID}.stockpulse.backtest_results"
)

SHORT_WINDOW = 20
LONG_WINDOW = 50

TRADING_DAYS = 252

# Illustrative cost of 0.05% per position change
TRANSACTION_COST = 0.0005

client = bigquery.Client(project=PROJECT_ID)

# ==========================================
# STEP 1: RETRIEVE STOCK PRICES
# ==========================================

query = f"""
SELECT
    ticker,
    date,
    close
FROM `{PRICES_TABLE}`
WHERE close > 0
ORDER BY ticker, date
"""

prices = client.query(query).to_dataframe()

if prices.empty:
    raise RuntimeError(
        "No price data found in BigQuery."
    )

prices["date"] = pd.to_datetime(prices["date"])

print("Successfully retrieved stock prices.")
print(prices.groupby("ticker").size())

results = []

# ==========================================
# STEP 2: BACKTEST EACH STOCK
# ==========================================

for ticker, df in prices.groupby("ticker"):

    print(f"\nBacktesting {ticker}...")

    df = df.sort_values("date").copy()
    df = df.drop_duplicates(
        subset=["date"], keep="last"
    )

    if len(df) < LONG_WINDOW + 2:
        print(f"Skipping {ticker}: insufficient data")
        continue

    # Moving averages
    df["sma20"] = (
        df["close"]
        .rolling(SHORT_WINDOW)
        .mean()
    )

    df["sma50"] = (
        df["close"]
        .rolling(LONG_WINDOW)
        .mean()
    )

    # Trading signal
    df["signal"] = (
        (df["sma20"] > df["sma50"])
        & df["sma50"].notna()
    ).astype(int)

    # Two-bar delay:
    # A signal at close t is reflected in
    # returns no earlier than t+2.
    # This conservatively avoids using the
    # signal on the day it is generated.
    df["position"] = (
        df["signal"].shift(2).fillna(0)
    )

    # Close-to-close daily returns
    df["asset_return"] = (
        df["close"].pct_change()
    )

    # Position changes
    df["turnover"] = (
        df["position"].diff().abs()
        .fillna(df["position"].abs())
    )

    # Strategy returns after illustrative costs
    df["strategy_return"] = (
        df["position"] * df["asset_return"]
        - TRANSACTION_COST * df["turnover"]
    )

    # Use dates after the moving-average warm-up
    valid = df[
        df["sma50"].notna()
        & df["asset_return"].notna()
    ].copy()

    if valid.empty:
        continue

    strategy_returns = valid["strategy_return"]
    buy_hold_returns = valid["asset_return"]

    # Equity curves
    strategy_equity = (
        1 + strategy_returns
    ).cumprod()

    buy_hold_equity = (
        1 + buy_hold_returns
    ).cumprod()

    # Total performance
    strategy_total = strategy_equity.iloc[-1] - 1
    buy_hold_total = buy_hold_equity.iloc[-1] - 1

    # Annualised volatility
    daily_std = strategy_returns.std()

    annual_vol = daily_std * np.sqrt(TRADING_DAYS)

    # Annualised Sharpe ratio
    # Zero risk-free rate assumed
    sharpe = (
        strategy_returns.mean()
        / daily_std
        * np.sqrt(TRADING_DAYS)
        if pd.notna(daily_std) and daily_std > 0
        else np.nan
    )

    # Maximum drawdown
    running_peak = np.maximum(
        strategy_equity.cummax(), 1.0
    )

    drawdown = (
        strategy_equity / running_peak - 1
    )

    max_drawdown = drawdown.min()

    # Number of entries/exits
    trade_events = int(
        (valid["turnover"] > 0).sum()
    )

    results.append({
        "ticker": ticker,
        "start_date": valid["date"].min().date(),
        "end_date": valid["date"].max().date(),
        "trading_days": len(valid),
        "strategy_return_pct": float(
            strategy_total * 100
        ),
        "buy_hold_return_pct": float(
            buy_hold_total * 100
        ),
        "annualized_volatility_pct": float(
            annual_vol * 100
        ),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(
            max_drawdown * 100
        ),
        "number_of_trades": trade_events
    })

# ==========================================
# STEP 3: CREATE SUMMARY
# ==========================================

report = pd.DataFrame(results)

if report.empty:
    raise RuntimeError(
        "No strategies could be evaluated."
    )

report = report.sort_values(
    "sharpe_ratio",
    ascending=False
)

print("\n========== BACKTEST RESULTS ==========")

print(
    report.round(3).to_string(index=False)
)

# ==========================================
# STEP 4: UPLOAD RESULTS TO BIGQUERY
# ==========================================

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(
    report,
    RESULTS_TABLE,
    job_config=job_config
)

job.result()

print(
    f"\nResults saved to BigQuery: "
    f"{RESULTS_TABLE}"
)
