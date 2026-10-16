import os
import pandas as pd
import yfinance as yf
from google.cloud import bigquery

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
DATASET_ID = "stockpulse"
TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.prices"

TICKERS = ["AAPL", "NVDA", "MSFT", "GOOGL", "SPY"]

frames = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start="2020-01-01",
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    df = df[
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()

    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.date

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    df["volume"] = (
        pd.to_numeric(df["volume"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    frames.append(df)

prices = pd.concat(frames, ignore_index=True)

client = bigquery.Client(project=PROJECT_ID)

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("open", "FLOAT64"),
        bigquery.SchemaField("high", "FLOAT64"),
        bigquery.SchemaField("low", "FLOAT64"),
        bigquery.SchemaField("close", "FLOAT64"),
        bigquery.SchemaField("volume", "INT64"),
        bigquery.SchemaField("ticker", "STRING"),
    ],
    write_disposition="WRITE_TRUNCATE",
)

job = client.load_table_from_dataframe(
    prices,
    TABLE_ID,
    job_config=job_config,
)
job.result()

print(f"Loaded {len(prices)} rows into {TABLE_ID}")
print(prices.groupby("ticker")["date"].agg(["min", "max", "count"]))

