CREATE OR REPLACE TABLE gold_recommender_user_index AS
WITH users AS (
  SELECT DISTINCT user_id
  FROM gold_user_item_splits
)
SELECT
  user_id,
  row_number() OVER (ORDER BY user_id) - 1 AS user_idx
FROM users
ORDER BY user_idx;
