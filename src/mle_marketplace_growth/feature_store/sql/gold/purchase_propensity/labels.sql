CREATE OR REPLACE TABLE gold_labels AS
WITH params AS (
  SELECT CAST('{as_of_date}' AS DATE) AS as_of_date
),
windows AS (
  SELECT 30 AS window_days
  UNION ALL
  SELECT 60 AS window_days
  UNION ALL
  SELECT 90 AS window_days
),
label_defs AS (
  SELECT 'net_revenue' AS metric
  UNION ALL
  SELECT 'purchase' AS metric
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
    windows.window_days,
    sum(silver_transactions_line_items.line_revenue) AS net_revenue,
    CASE WHEN count(*) > 0 THEN 1.0 ELSE 0.0 END AS purchase
  FROM silver_transactions_line_items, params, windows
  WHERE silver_transactions_line_items.user_id <> ''
    AND silver_transactions_line_items.quantity > 0
    AND silver_transactions_line_items.event_date > params.as_of_date
    AND silver_transactions_line_items.event_date <= params.as_of_date + windows.window_days * INTERVAL 1 DAY
  GROUP BY silver_transactions_line_items.user_id, windows.window_days
)
SELECT
  users.user_id,
  params.as_of_date AS as_of_date,
  concat(label_defs.metric, '_', windows.window_days, 'd') AS label_name,
  CASE
    WHEN label_defs.metric = 'net_revenue' THEN coalesce(future_purchases.net_revenue, 0.0)
    ELSE coalesce(future_purchases.purchase, 0.0)
  END AS label_value,
  windows.window_days AS window_days
FROM users
CROSS JOIN params
CROSS JOIN windows
CROSS JOIN label_defs
LEFT JOIN future_purchases
  ON users.user_id = future_purchases.user_id
 AND windows.window_days = future_purchases.window_days
ORDER BY users.user_id, windows.window_days, label_defs.metric;
