CREATE OR REPLACE TABLE gold_labels AS
WITH params AS (
  SELECT
    CAST('{as_of_date}' AS DATE) AS as_of_date,
    CAST('{as_of_date}' AS DATE) + INTERVAL 30 DAY AS label_end_date
),
label_names AS (
  SELECT 'net_revenue_30d' AS label_name
  UNION ALL
  SELECT 'purchase_30d' AS label_name
),
users AS (
  SELECT DISTINCT user_id
  FROM silver_transactions_line_items, params
  WHERE user_id <> ''
    AND event_date <= params.as_of_date
),
future_purchases AS (
  SELECT
    silver_transactions_line_items.user_id,
    sum(silver_transactions_line_items.line_revenue) AS net_revenue_30d,
    CASE WHEN count(*) > 0 THEN 1.0 ELSE 0.0 END AS purchase_30d
  FROM silver_transactions_line_items, params
  WHERE silver_transactions_line_items.user_id <> ''
    AND silver_transactions_line_items.quantity > 0
    AND silver_transactions_line_items.event_date > params.as_of_date
    AND silver_transactions_line_items.event_date <= params.label_end_date
  GROUP BY silver_transactions_line_items.user_id
)
SELECT
  users.user_id,
  params.as_of_date AS as_of_date,
  label_names.label_name,
  CASE
    WHEN label_names.label_name = 'net_revenue_30d' THEN coalesce(future_purchases.net_revenue_30d, 0.0)
    ELSE coalesce(future_purchases.purchase_30d, 0.0)
  END AS label_value,
  30 AS window_days
FROM users
CROSS JOIN params
CROSS JOIN label_names
LEFT JOIN future_purchases ON users.user_id = future_purchases.user_id
ORDER BY users.user_id, label_names.label_name;
