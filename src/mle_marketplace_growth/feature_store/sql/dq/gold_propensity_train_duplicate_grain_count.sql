SELECT COUNT(*)
FROM (
  SELECT user_id, as_of_date, COUNT(*) AS c
  FROM gold_propensity_train_dataset
  GROUP BY user_id, as_of_date
  HAVING c > 1
);
