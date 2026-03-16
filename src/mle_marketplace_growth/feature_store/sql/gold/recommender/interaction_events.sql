-- Purpose: Build gold_interaction_events for recommender training/eval.
-- Runtime inputs:
-- - {recommender_min_event_date}
-- - {recommender_max_event_date}
-- Note: the gold table retains `weight = quantity`, but the current recommender
-- training path uses binary user-item interactions only (each unique pair counts once).
CREATE OR REPLACE TABLE gold_interaction_events AS
-- Select interaction events for recommender training/eval.
SELECT
  user_id,
  item_id,
  invoice_id,
  STRFTIME(event_ts, '%Y-%m-%d %H:%M:%S') AS event_ts,
  CAST(event_date AS VARCHAR) AS event_date,
  quantity AS weight
FROM silver_transactions_line_items
WHERE quantity > 0
  AND user_id <> ''
  AND event_date >= CAST('{recommender_min_event_date}' AS DATE)
  AND event_date <= CAST('{recommender_max_event_date}' AS DATE)
ORDER BY event_ts, user_id, item_id;
