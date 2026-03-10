-- Purpose: Build gold_propensity_train_dataset for one as_of_date snapshot (single date per run).
-- Notes (scope: SQL vs training/serving):
-- 1) SQL: 30d short-term lookback windows (frequency_30d, monetary_30d) are present.
-- 2) SQL: longer-term lookback windows (frequency_*, monetary_*, avg_basket_value_*) are present; model selects a profile.
-- 3) Training/serving: longer-term monetary features are capped to reduce skew (not applied in SQL).
-- 4) SQL: avg_basket_value_30d is omitted; 30d counts/sums remain for short-term signal.
-- 5) SQL: label_* columns are present for multiple horizons; model uses configured prediction window.
-- See docs/purchase_propensity/spec.md for the rationale behind these choices.
CREATE OR REPLACE TABLE gold_propensity_train_dataset AS
WITH
labels_pivot AS (
  -- Pivot labels to one row per user/as_of_date.
  SELECT
    user_id,
    as_of_date,
    MAX(CASE WHEN label_name = 'net_revenue_60d' THEN label_value END) AS label_net_revenue_60d,
    MAX(CASE WHEN label_name = 'net_revenue_90d' THEN label_value END) AS label_net_revenue_90d,
    MAX(CASE WHEN label_name = 'net_revenue_30d' THEN label_value END) AS label_net_revenue_30d,
    MAX(CASE WHEN label_name = 'purchase_60d' THEN label_value END) AS label_purchase_60d,
    MAX(CASE WHEN label_name = 'purchase_90d' THEN label_value END) AS label_purchase_90d,
    MAX(CASE WHEN label_name = 'purchase_30d' THEN label_value END) AS label_purchase_30d
  FROM gold_labels
  WHERE as_of_date = CAST('{as_of_date}' AS DATE)
  GROUP BY user_id, as_of_date
)
-- Select features joined with pivoted labels for training.
SELECT
  features.user_id,
  features.as_of_date,
  features.recency_days,
  features.frequency_30d,
  features.frequency_60d,
  features.frequency_90d,
  features.frequency_120d,
  features.monetary_30d,
  features.monetary_60d,
  features.monetary_90d,
  features.monetary_120d,
  features.avg_basket_value_60d,
  features.avg_basket_value_90d,
  features.avg_basket_value_120d,
  features.country,
  COALESCE(labels_pivot.label_net_revenue_60d, 0.0) AS label_net_revenue_60d,
  COALESCE(labels_pivot.label_net_revenue_90d, 0.0) AS label_net_revenue_90d,
  COALESCE(labels_pivot.label_net_revenue_30d, 0.0) AS label_net_revenue_30d,
  COALESCE(labels_pivot.label_purchase_60d, 0.0) AS label_purchase_60d,
  COALESCE(labels_pivot.label_purchase_90d, 0.0) AS label_purchase_90d,
  COALESCE(labels_pivot.label_purchase_30d, 0.0) AS label_purchase_30d
FROM gold_user_features_asof AS features
LEFT JOIN labels_pivot
  ON features.user_id = labels_pivot.user_id
 AND features.as_of_date = labels_pivot.as_of_date
WHERE features.as_of_date = CAST('{as_of_date}' AS DATE)
ORDER BY features.user_id;
