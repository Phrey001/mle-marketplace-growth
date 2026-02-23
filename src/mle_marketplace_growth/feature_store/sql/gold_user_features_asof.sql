CREATE OR REPLACE TABLE gold_user_features_asof AS
WITH params AS (
  SELECT CAST('{as_of_date}' AS DATE) AS as_of_date
),
all_events AS (
  SELECT
    user_id,
    invoice_id,
    event_ts,
    event_date,
    line_revenue,
    country,
    quantity
  FROM silver_transactions_line_items, params
  WHERE user_id <> ''
    AND event_date <= params.as_of_date
),
positive_events AS (
  SELECT
    user_id,
    invoice_id,
    event_ts,
    event_date,
    line_revenue,
    country
  FROM all_events
  WHERE quantity > 0
),
users AS (
  SELECT DISTINCT user_id FROM all_events
),
latest AS (
  SELECT
    user_id,
    max(event_ts) AS last_event_ts,
    arg_max(country, event_ts) AS country
  FROM positive_events
  GROUP BY user_id
),
user_rollups AS (
  SELECT
    positive_events.user_id,
    count(DISTINCT CASE WHEN positive_events.event_date >= params.as_of_date - INTERVAL 29 DAY THEN positive_events.invoice_id END) AS frequency_30d,
    count(DISTINCT CASE WHEN positive_events.event_date >= params.as_of_date - INTERVAL 89 DAY THEN positive_events.invoice_id END) AS frequency_90d,
    sum(CASE WHEN positive_events.event_date >= params.as_of_date - INTERVAL 29 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_30d,
    sum(CASE WHEN positive_events.event_date >= params.as_of_date - INTERVAL 89 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_90d
  FROM positive_events, params
  GROUP BY positive_events.user_id
),
invoice_totals_90d AS (
  SELECT
    positive_events.user_id,
    positive_events.invoice_id,
    sum(positive_events.line_revenue) AS invoice_revenue
  FROM positive_events, params
  WHERE positive_events.event_date >= params.as_of_date - INTERVAL 89 DAY
  GROUP BY positive_events.user_id, positive_events.invoice_id
),
basket_90d AS (
  SELECT
    user_id,
    avg(invoice_revenue) AS avg_basket_value_90d
  FROM invoice_totals_90d
  GROUP BY user_id
)
SELECT
  users.user_id,
  params.as_of_date AS as_of_date,
  coalesce(date_diff('day', cast(latest.last_event_ts AS DATE), params.as_of_date), 9999) AS recency_days,
  coalesce(user_rollups.frequency_30d, 0) AS frequency_30d,
  coalesce(user_rollups.frequency_90d, 0) AS frequency_90d,
  round(coalesce(user_rollups.monetary_30d, 0), 4) AS monetary_30d,
  round(coalesce(user_rollups.monetary_90d, 0), 4) AS monetary_90d,
  round(coalesce(basket_90d.avg_basket_value_90d, 0), 4) AS avg_basket_value_90d,
  coalesce(latest.country, 'UNKNOWN') AS country
FROM users
CROSS JOIN params
LEFT JOIN latest ON users.user_id = latest.user_id
LEFT JOIN user_rollups ON users.user_id = user_rollups.user_id
LEFT JOIN basket_90d ON users.user_id = basket_90d.user_id
ORDER BY users.user_id;
