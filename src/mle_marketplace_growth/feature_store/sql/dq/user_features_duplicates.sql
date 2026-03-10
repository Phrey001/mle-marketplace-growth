-- Purpose: Count duplicate (user_id, as_of_date) rows in gold_user_features_asof.
-- Select count of duplicate user/as_of_date rows.
SELECT COUNT(*)
FROM (
  -- Select user/as_of_date groups with more than one row.
  SELECT user_id, as_of_date, COUNT(*) AS c
  FROM gold_user_features_asof
  GROUP BY user_id, as_of_date
  HAVING c > 1
);
