-- Purpose: Count gold_user_item_splits rows with invalid split or split_version.
-- Select count of invalid split rows.
SELECT COUNT(*)
FROM gold_user_item_splits
WHERE split NOT IN ('train', 'val', 'test')
   OR split_version = '';
