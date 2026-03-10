-- Purpose: Assign stable item indices from train split in gold_user_item_splits.
CREATE OR REPLACE TABLE gold_recommender_item_index AS
WITH
train_items AS (
  -- Select items observed in the train split for index assignment.
  SELECT DISTINCT item_id
  FROM gold_user_item_splits
  WHERE split = 'train'
)
-- Select item index mapping.
SELECT
  item_id,
  ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_idx
FROM train_items
ORDER BY item_idx;
