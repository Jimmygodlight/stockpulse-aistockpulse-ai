-- STOCKPULSE AI | STEP 19.2: FEATURE ENGINEERING
-- Run this first in BigQuery Studio.
-- Creates/overwrites only stockpulse.stock_features; does not alter stockpulse.prices.
-- Price features use observations through the close on date t.
-- Label = direction of the next observed trading day's close (t+1).
-- Requires one row per ticker/date in the source table. Check duplicates separately.

CREATE OR REPLACE TABLE `stockpulse-ai-511816.stockpulse.stock_features` AS
WITH base AS (
  SELECT
    ticker,
    date,
    close,
    volume,
    LAG(close, 1) OVER w AS prev_close,
    LAG(close, 5) OVER w AS close_5d_ago,
    LAG(close, 20) OVER w AS close_20d_ago,
    LEAD(close, 1) OVER w AS next_close,
    AVG(close) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS sma20,
    AVG(close) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ) AS sma50,
    AVG(volume) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS avg_volume20,
    COUNT(*) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
    ) AS window_count
  FROM `stockpulse-ai-511816.stockpulse.prices`
  WHERE close > 0
  WINDOW w AS (PARTITION BY ticker ORDER BY date)
),
returns AS (
  SELECT *,
    SAFE_DIVIDE(close, prev_close) - 1 AS ret_1d
  FROM base
),
features AS (
  SELECT *,
    STDDEV_SAMP(ret_1d) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS volatility_20d
  FROM returns
)
SELECT
  ticker,
  date,
  ret_1d,
  SAFE_DIVIDE(close, close_5d_ago) - 1 AS ret_5d,
  SAFE_DIVIDE(close, close_20d_ago) - 1 AS ret_20d,
  volatility_20d,
  SAFE_DIVIDE(sma20, sma50) - 1 AS sma_gap_20_50,
  SAFE_DIVIDE(volume, avg_volume20) - 1 AS relative_volume,
  CAST(next_close > close AS INT64) AS up_next_day,
  CASE
    WHEN date <= DATE '2023-11-30' THEN 'train'
    WHEN date BETWEEN DATE '2024-01-01' AND DATE '2024-11-30'
      THEN 'validation'
    WHEN date >= DATE '2025-01-01' THEN 'test'
    ELSE 'unused'
  END AS sample_group
FROM features
WHERE window_count = 50
  AND next_close IS NOT NULL
  AND volatility_20d IS NOT NULL
  AND avg_volume20 > 0
  AND prev_close > 0
  AND close_5d_ago > 0
  AND close_20d_ago > 0;

-- Check that all three partitions exist and dates are sensible.
SELECT
  sample_group,
  COUNT(*) AS observations,
  COUNT(DISTINCT ticker) AS tickers,
  MIN(date) AS first_date,
  MAX(date) AS last_date,
  ROUND(AVG(up_next_day), 4) AS fraction_up
FROM `stockpulse-ai-511816.stockpulse.stock_features`
GROUP BY sample_group
ORDER BY sample_group;

-- Additional source-data integrity check (expect ZERO rows).
SELECT ticker, date, COUNT(*) AS repeated_rows
FROM `stockpulse-ai-511816.stockpulse.prices`
GROUP BY ticker, date
HAVING COUNT(*) > 1
ORDER BY repeated_rows DESC
LIMIT 20;
