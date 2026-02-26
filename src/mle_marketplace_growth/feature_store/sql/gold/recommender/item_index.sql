CREATE OR REPLACE TABLE gold_recommender_item_index AS
WITH train_items AS (
  SELECT DISTINCT item_id
  FROM gold_user_item_splits
  WHERE split = 'train'
)
SELECT
  item_id,
  row_number() OVER (ORDER BY item_id) - 1 AS item_idx
FROM train_items
ORDER BY item_idx;
