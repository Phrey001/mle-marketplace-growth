-- Purpose: Count gold_user_item_splits rows with invalid split values.
-- Why: Keeps the recommender split contract limited to the expected train/val/test values.
-- Select count of invalid split rows.
SELECT COUNT(*)
FROM gold_user_item_splits
WHERE split NOT IN ('train', 'val', 'test');
