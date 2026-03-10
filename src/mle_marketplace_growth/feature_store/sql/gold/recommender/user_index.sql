-- Purpose: Assign stable user indices from gold_user_item_splits.
CREATE OR REPLACE TABLE gold_recommender_user_index AS
WITH
users AS (
  -- Select unique users for index assignment.
  SELECT DISTINCT user_id
  FROM gold_user_item_splits
)
-- Select user index mapping.
SELECT
  user_id,
  ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_idx
FROM users
ORDER BY user_idx;
