-- Purpose: Assign stable user indices from gold_user_item_splits.
-- Why: Matrix/embedding models need deterministic user row indices shared across train/eval/serve.
CREATE OR REPLACE TABLE gold_recommender_user_index AS
WITH
users AS (
  -- Keep one row per recommender user before assigning a stable row index.
  SELECT DISTINCT user_id
  FROM gold_user_item_splits
)
-- Assign one deterministic zero-based row index per user.
SELECT
  user_id,
  ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_idx
FROM users
ORDER BY user_idx;
