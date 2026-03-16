-- Purpose: Assign stable item indices from train split in gold_user_item_splits.
-- Why: Matrix/embedding models need deterministic row indices for the train item universe.
CREATE OR REPLACE TABLE gold_recommender_item_index AS
WITH
train_items AS (
  -- Limit indices to train-split items so retrieval only scores the train item universe.
  SELECT DISTINCT item_id
  FROM gold_user_item_splits
  WHERE split = 'train'
)
-- Assign one deterministic zero-based row index per train item.
SELECT
  item_id,
  ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_idx
FROM train_items
ORDER BY item_idx;
