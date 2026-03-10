-- Purpose: Build gold_user_features_asof for one as_of_date snapshot (single date per run).
-- Note: avg_basket_value_* features are average invoice values (1 basket = 1 purchase invoice).
-- Note: see docs/purchase_propensity/spec.md for feature window selection details.
CREATE OR REPLACE TABLE gold_user_features_asof AS
WITH
as_of_anchor AS (
  -- Select anchor date for snapshot feature computation.
  SELECT CAST('{as_of_date}' AS DATE) AS as_of_date
),
all_events AS (
  -- Select all events up to snapshot (includes returns).
  SELECT
    user_id,
    invoice_id,
    event_ts,
    event_date,
    line_revenue,
    country,
    quantity
  FROM silver_transactions_line_items
  CROSS JOIN as_of_anchor
  WHERE user_id <> ''
    AND event_date <= as_of_anchor.as_of_date
),
positive_events AS (
  -- Select positive-quantity events used for rollups.
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
  -- Select distinct users present in the snapshot window.
  SELECT DISTINCT user_id FROM all_events
),
latest_ranked AS (
  -- Rank events per user so we can pick the most recent row using standard SQL.
  SELECT
    user_id,
    event_ts,
    country,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY event_ts DESC
    ) AS recency_rank
  FROM positive_events
),
latest AS (
  -- Select most recent interaction timestamp + country per user (portable form).
  SELECT
    user_id,
    event_ts AS last_event_ts,
    country
  FROM latest_ranked
  WHERE recency_rank = 1
),
user_rollups AS (
  -- Aggregate lookback frequency and monetary features over multiple windows.
  SELECT
    positive_events.user_id,
    COUNT(DISTINCT CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 29 DAY THEN positive_events.invoice_id END) AS frequency_30d,
    COUNT(DISTINCT CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 59 DAY THEN positive_events.invoice_id END) AS frequency_60d,
    COUNT(DISTINCT CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 89 DAY THEN positive_events.invoice_id END) AS frequency_90d,
    COUNT(DISTINCT CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 119 DAY THEN positive_events.invoice_id END) AS frequency_120d,
    SUM(CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 29 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_30d,
    SUM(CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 59 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_60d,
    SUM(CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 89 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_90d,
    SUM(CASE WHEN positive_events.event_date >= as_of_anchor.as_of_date - INTERVAL 119 DAY THEN positive_events.line_revenue ELSE 0 END) AS monetary_120d
  FROM positive_events
  CROSS JOIN as_of_anchor
  GROUP BY positive_events.user_id
),
invoice_windows AS (
  -- Select invoice-value lookback windows.
  SELECT 60 AS lookback_window_days
  UNION ALL
  SELECT 90 AS lookback_window_days
  UNION ALL
  SELECT 120 AS lookback_window_days
),
invoice_totals_windowed AS (
  -- Aggregate to invoice-level totals per user and window before averaging.
  SELECT
    positive_events.user_id,
    positive_events.invoice_id,
    invoice_windows.lookback_window_days,
    SUM(positive_events.line_revenue) AS invoice_revenue
  FROM positive_events
  CROSS JOIN as_of_anchor
  CROSS JOIN invoice_windows
  -- INTERVAL turns window_days into a day interval; minus 1 keeps the lookback inclusive of as_of_date.
  WHERE positive_events.event_date >= as_of_anchor.as_of_date - (invoice_windows.lookback_window_days - 1) * INTERVAL 1 DAY
  GROUP BY positive_events.user_id, positive_events.invoice_id, invoice_windows.lookback_window_days
),
invoice_avg_by_window AS (
  -- Compute average invoice total per user and window (basket value).
  SELECT
    user_id,
    lookback_window_days,
    AVG(invoice_revenue) AS avg_basket_value
  FROM invoice_totals_windowed
  GROUP BY user_id, lookback_window_days
),
invoice_avg_pivot AS (
  -- Pivot invoice-average values into wide columns.
  SELECT
    user_id,
    MAX(CASE WHEN lookback_window_days = 60 THEN avg_basket_value END) AS avg_basket_value_60d,
    MAX(CASE WHEN lookback_window_days = 90 THEN avg_basket_value END) AS avg_basket_value_90d,
    MAX(CASE WHEN lookback_window_days = 120 THEN avg_basket_value END) AS avg_basket_value_120d
  FROM invoice_avg_by_window
  GROUP BY user_id
)
-- Select final per-user snapshot features.
SELECT
  users.user_id,
  as_of_anchor.as_of_date AS as_of_date,
  -- If no prior events, use 9999 so recency is much larger than any real user;
  -- downstream models learn that very large recency = no recent purchase history.
  COALESCE(DATE_DIFF('day', CAST(latest.last_event_ts AS DATE), as_of_anchor.as_of_date), 9999) AS recency_days,
  COALESCE(user_rollups.frequency_30d, 0) AS frequency_30d,
  COALESCE(user_rollups.frequency_60d, 0) AS frequency_60d,
  COALESCE(user_rollups.frequency_90d, 0) AS frequency_90d,
  COALESCE(user_rollups.frequency_120d, 0) AS frequency_120d,
  ROUND(COALESCE(user_rollups.monetary_30d, 0), 4) AS monetary_30d,
  ROUND(COALESCE(user_rollups.monetary_60d, 0), 4) AS monetary_60d,
  ROUND(COALESCE(user_rollups.monetary_90d, 0), 4) AS monetary_90d,
  ROUND(COALESCE(user_rollups.monetary_120d, 0), 4) AS monetary_120d,
  ROUND(COALESCE(invoice_avg_pivot.avg_basket_value_60d, 0), 4) AS avg_basket_value_60d,
  ROUND(COALESCE(invoice_avg_pivot.avg_basket_value_90d, 0), 4) AS avg_basket_value_90d,
  ROUND(COALESCE(invoice_avg_pivot.avg_basket_value_120d, 0), 4) AS avg_basket_value_120d,
  COALESCE(latest.country, 'UNKNOWN') AS country
FROM users
CROSS JOIN as_of_anchor
LEFT JOIN latest ON users.user_id = latest.user_id
LEFT JOIN user_rollups ON users.user_id = user_rollups.user_id
LEFT JOIN invoice_avg_pivot ON users.user_id = invoice_avg_pivot.user_id
ORDER BY users.user_id;
