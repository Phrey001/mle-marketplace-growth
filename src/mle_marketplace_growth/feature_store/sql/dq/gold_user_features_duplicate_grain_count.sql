SELECT COUNT(*)
FROM (
  SELECT user_id, as_of_date, COUNT(*) AS c
  FROM gold_user_features_asof
  GROUP BY user_id, as_of_date
  HAVING c > 1
);
