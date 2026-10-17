-- STOCKPULSE AI | STEP 19.6: TRAIN BOOSTED-TREE CLASSIFIER
-- Run AFTER features.sql. May cost more than logistic regression.
-- Same features and training partition as logistic.sql, for fair comparison.

CREATE OR REPLACE MODEL `stockpulse-ai-511816.stockpulse.direction_boosted`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['up_next_day'],
  data_split_method = 'NO_SPLIT',
  max_iterations = 25,
  max_tree_depth = 3,
  early_stop = FALSE
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
