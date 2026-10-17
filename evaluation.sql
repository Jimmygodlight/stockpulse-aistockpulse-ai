-- STOCKPULSE AI | STEP 19.5-19.6: VALIDATION EVALUATION
-- Run AFTER BOTH logistic.sql and boosted.sql have succeeded.
-- Choose the better model on VALIDATION, then run a final TEST only once.
-- Do not tune the models against the final TEST observations.

-- 1) Naive majority-class benchmark, for validation only.
SELECT
  COUNT(*) AS validation_observations,
  ROUND(AVG(up_next_day), 4) AS fraction_up,
  ROUND(GREATEST(AVG(up_next_day), 1 - AVG(up_next_day)), 4)
    AS majority_class_accuracy
FROM `stockpulse-ai-511816.stockpulse.stock_features`
WHERE sample_group = 'validation';

-- 2) Compare both trained classifiers on SAME validation rows.
SELECT
  'logistic' AS model_name,
  accuracy, precision, recall, f1_score, log_loss, roc_auc
FROM ML.EVALUATE(
  MODEL `stockpulse-ai-511816.stockpulse.direction_logistic`,
  (
    SELECT
      ticker, ret_1d, ret_5d, ret_20d, volatility_20d,
      sma_gap_20_50, relative_volume, up_next_day
    FROM `stockpulse-ai-511816.stockpulse.stock_features`
    WHERE sample_group = 'validation'
  )
)
UNION ALL
SELECT
  'boosted' AS model_name,
  accuracy, precision, recall, f1_score, log_loss, roc_auc
FROM ML.EVALUATE(
  MODEL `stockpulse-ai-511816.stockpulse.direction_boosted`,
  (
    SELECT
      ticker, ret_1d, ret_5d, ret_20d, volatility_20d,
      sma_gap_20_50, relative_volume, up_next_day
    FROM `stockpulse-ai-511816.stockpulse.stock_features`
    WHERE sample_group = 'validation'
  )
)
ORDER BY model_name;

-- 3) FINAL HOLDOUT TEST (intentionally commented out).
-- Once you have selected ONE model using validation results, copy the
-- query below into a NEW BigQuery query tab, remove the comment markers,
-- and replace 'direction_logistic' with 'direction_boosted' if selected.
-- Run it only once after model selection. Save the metrics to evaluation/.
--
-- SELECT *
-- FROM ML.EVALUATE(
--   MODEL `stockpulse-ai-511816.stockpulse.direction_logistic`,
--   (
--     SELECT
--       ticker, ret_1d, ret_5d, ret_20d, volatility_20d,
--       sma_gap_20_50, relative_volume, up_next_day
--     FROM `stockpulse-ai-511816.stockpulse.stock_features`
--     WHERE sample_group = 'test'
--   )
-- );
