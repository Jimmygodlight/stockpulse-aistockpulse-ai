-- STOCKPULSE AI | STEP 19.4: TRAIN LOGISTIC REGRESSION
-- Run AFTER features.sql.
-- This CREATE OR REPLACE MODEL job can incur BigQuery ML charges.
-- NO_SPLIT means all rows selected here are used for training;
-- validation and test are evaluated separately in evaluation.sql.

CREATE OR REPLACE MODEL `stockpulse-ai-511816.stockpulse.direction_logistic`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['up_next_day'],
  data_split_method = 'NO_SPLIT',
  max_iterations = 30
)
AS
SELECT
  ticker,
  ret_1d,
  ret_5d,
  ret_20d,
  volatility_20d,
  sma_gap_20_50,
  relative_volume,
  up_next_day
FROM `stockpulse-ai-511816.stockpulse.stock_features`
WHERE sample_group = 'train';
