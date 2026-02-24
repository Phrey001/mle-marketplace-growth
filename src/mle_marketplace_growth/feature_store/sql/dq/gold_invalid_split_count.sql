SELECT COUNT(*)
FROM gold_user_item_splits
WHERE split NOT IN ('train', 'val', 'test')
   OR split_version = '';
