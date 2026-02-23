CREATE OR REPLACE TABLE gold_interaction_events AS
SELECT
  user_id,
  item_id,
  invoice_id,
  strftime(event_ts, '%Y-%m-%d %H:%M:%S') AS event_ts,
  cast(event_date AS VARCHAR) AS event_date,
  quantity AS weight
FROM silver_transactions_line_items
WHERE quantity > 0
  AND user_id <> ''
ORDER BY event_ts, user_id, item_id;

