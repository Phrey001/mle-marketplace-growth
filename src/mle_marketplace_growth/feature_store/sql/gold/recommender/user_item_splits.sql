CREATE OR REPLACE TABLE gold_user_item_splits AS
WITH ordered_interactions AS (
  SELECT
    user_id,
    item_id,
    invoice_id,
    event_ts,
    event_date,
    weight,
    row_number() OVER (
      PARTITION BY user_id
      ORDER BY event_ts DESC, invoice_id DESC, item_id DESC
    ) AS recency_rank
  FROM gold_interaction_events
)
SELECT
  user_id,
  item_id,
  invoice_id,
  event_ts,
  event_date,
  weight,
  'time_rank_v1' AS split_version,
  CASE
    WHEN recency_rank = 1 THEN 'test'
    WHEN recency_rank = 2 THEN 'val'
    ELSE 'train'
  END AS split
FROM ordered_interactions
ORDER BY event_ts, user_id, item_id;
