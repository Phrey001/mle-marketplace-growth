-- Purpose: Build gold_interaction_events for recommender training/eval.
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
  {recommender_time_filters}
ORDER BY event_ts, user_id, item_id;
