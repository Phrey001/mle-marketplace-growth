WITH split_times AS (
  SELECT
    user_id,
    max(CASE WHEN split = 'train' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS max_train_ts,
    min(CASE WHEN split = 'val' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_val_ts,
    min(CASE WHEN split = 'test' THEN try_strptime(event_ts, '%Y-%m-%d %H:%M:%S') END) AS min_test_ts
  FROM gold_user_item_splits
  GROUP BY user_id
)
SELECT COUNT(*)
FROM split_times
WHERE (max_train_ts IS NOT NULL AND min_val_ts IS NOT NULL AND max_train_ts > min_val_ts)
   OR (min_val_ts IS NOT NULL AND min_test_ts IS NOT NULL AND min_val_ts > min_test_ts);
