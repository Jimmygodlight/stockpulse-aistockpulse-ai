-- Step 19: TEST-period baseline, using the majority class decided on training data.
-- Read-only query; does not retrain or modify models.
WITH training AS (
  SELECT AVG(CAST(up_next_day AS FLOAT64)) AS p_up
  FROM `stockpulse-ai-511816.stockpulse.stock_features`
  WHERE sample_group = 'train'
),
testing AS (
  SELECT CAST(up_next_day AS INT64) AS up_next_day
  FROM `stockpulse-ai-511816.stockpulse.stock_features`
  WHERE sample_group = 'test'
)
SELECT
  COUNT(*) AS test_observations,
  ROUND(100 * AVG(CAST(t.up_next_day AS FLOAT64)), 2) AS actual_test_up_days_pct,
  IF(training.p_up >= 0.5, 'Always predict UP', 'Always predict DOWN') AS baseline_strategy,
  ROUND(100 * AVG(IF(t.up_next_day = IF(training.p_up >= 0.5, 1, 0), 1.0, 0.0)), 2)
    AS baseline_test_accuracy_pct,
  ROUND(AVG(IF(t.up_next_day = 1,
    -LN(GREATEST(training.p_up, 0.000001)),
    -LN(GREATEST(1.0 - training.p_up, 0.000001)))), 4)
    AS baseline_test_log_loss
FROM testing AS t
CROSS JOIN training
GROUP BY training.p_up;
