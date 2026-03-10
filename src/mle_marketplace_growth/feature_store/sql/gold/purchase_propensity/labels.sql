-- Purpose: Build gold_labels for one as_of_date snapshot (single date per run).
-- Note: labels are created for multiple future horizons; models pick the configured prediction window.
-- Note: see docs/purchase_propensity/spec.md for window selection details.
CREATE OR REPLACE TABLE gold_labels AS
WITH
as_of_anchor AS (
  -- Select anchor date for labeling windows.
  SELECT CAST('{as_of_date}' AS DATE) AS as_of_date
),
future_purchase_windows AS (
  -- Select future prediction horizons (post as_of_date) to generate labels.
  SELECT 30 AS window_days
  UNION ALL
  SELECT 60 AS window_days
  UNION ALL
  SELECT 90 AS window_days
),
label_types AS (
  -- Select label types produced for each horizon.
  SELECT 'net_revenue' AS metric
  UNION ALL
  SELECT 'purchase' AS metric
),
users AS (
  -- Select users observed up to the as-of snapshot.
  SELECT DISTINCT user_id
  FROM silver_transactions_line_items
  CROSS JOIN as_of_anchor
  WHERE user_id <> ''
    AND event_date <= as_of_anchor.as_of_date
),
future_purchases AS (
  -- Aggregate outcomes after the snapshot over each horizon.
  SELECT
    silver_transactions_line_items.user_id,
    future_purchase_windows.window_days,
    SUM(silver_transactions_line_items.line_revenue) AS net_revenue,
    CASE WHEN COUNT(*) > 0 THEN 1.0 ELSE 0.0 END AS purchase_flag
  FROM silver_transactions_line_items
  CROSS JOIN as_of_anchor
  CROSS JOIN future_purchase_windows
  WHERE silver_transactions_line_items.user_id <> ''
    AND silver_transactions_line_items.quantity > 0
    AND silver_transactions_line_items.event_date > as_of_anchor.as_of_date
    -- INTERVAL turns the integer window_days into a day interval for DATE arithmetic.
    AND silver_transactions_line_items.event_date <= as_of_anchor.as_of_date + future_purchase_windows.window_days * INTERVAL 1 DAY
  GROUP BY silver_transactions_line_items.user_id, future_purchase_windows.window_days
)
-- Select labeled outcomes per user, horizon, and label type.
SELECT
  users.user_id,
  as_of_anchor.as_of_date AS as_of_date,
  concat(label_types.metric, '_', future_purchase_windows.window_days, 'd') AS label_name,
  future_purchase_windows.window_days AS window_days,
  -- Note: labels are forward-looking outcomes built from historical data.
  -- Many label_value entries are zero when no purchase occurs in the horizon.
  CASE
    WHEN label_types.metric = 'net_revenue' THEN COALESCE(future_purchases.net_revenue, 0.0)
    WHEN label_types.metric = 'purchase' THEN COALESCE(future_purchases.purchase_flag, 0.0)
    ELSE NULL
  END AS label_value
FROM users
-- CROSS JOINs create all (user, horizon, label_type) combinations for labeling.
CROSS JOIN as_of_anchor
CROSS JOIN future_purchase_windows
CROSS JOIN label_types
LEFT JOIN future_purchases
  ON users.user_id = future_purchases.user_id
 AND future_purchase_windows.window_days = future_purchases.window_days
ORDER BY users.user_id, future_purchase_windows.window_days, label_types.metric;
