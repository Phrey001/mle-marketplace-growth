-- Purpose: Count users with train/val/test chronology violations in gold_user_item_splits.
-- Why: Recommender offline evaluation must keep older interactions in train and newer ones in val/test.
WITH
split_times AS (
  -- Collapse each user's split timestamps so we can compare the latest train event
  -- against the earliest val/test events for chronology validation.
  SELECT
    user_id,
    MAX(CASE WHEN split = 'train' THEN TRY_STRPTIME(event_ts, '%Y-%m-%d %H:%M:%S') END) AS max_train_ts,
    MIN(CASE WHEN split = 'val' THEN TRY_STRPTIME(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_val_ts,
    MIN(CASE WHEN split = 'test' THEN TRY_STRPTIME(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_test_ts
  FROM gold_user_item_splits
  GROUP BY user_id
)
-- Count users where the configured chronological split contract is broken.
SELECT COUNT(*)
FROM split_times
WHERE (max_train_ts IS NOT NULL AND min_val_ts IS NOT NULL AND max_train_ts > min_val_ts)
   OR (min_val_ts IS NOT NULL AND min_test_ts IS NOT NULL AND min_val_ts > min_test_ts);
