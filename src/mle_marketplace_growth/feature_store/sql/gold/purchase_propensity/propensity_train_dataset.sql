CREATE OR REPLACE TABLE gold_propensity_train_dataset AS
WITH labels_pivot AS (
  SELECT
    user_id,
    as_of_date,
    max(CASE WHEN label_name = 'net_revenue_60d' THEN label_value END) AS label_net_revenue_60d,
    max(CASE WHEN label_name = 'net_revenue_90d' THEN label_value END) AS label_net_revenue_90d,
    max(CASE WHEN label_name = 'net_revenue_30d' THEN label_value END) AS label_net_revenue_30d,
    max(CASE WHEN label_name = 'purchase_60d' THEN label_value END) AS label_purchase_60d,
    max(CASE WHEN label_name = 'purchase_90d' THEN label_value END) AS label_purchase_90d,
    max(CASE WHEN label_name = 'purchase_30d' THEN label_value END) AS label_purchase_30d
  FROM gold_labels
  WHERE as_of_date = CAST('{as_of_date}' AS DATE)
  GROUP BY user_id, as_of_date
)
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
  coalesce(labels_pivot.label_net_revenue_60d, 0.0) AS label_net_revenue_60d,
  coalesce(labels_pivot.label_net_revenue_90d, 0.0) AS label_net_revenue_90d,
  coalesce(labels_pivot.label_net_revenue_30d, 0.0) AS label_net_revenue_30d,
  coalesce(labels_pivot.label_purchase_60d, 0.0) AS label_purchase_60d,
  coalesce(labels_pivot.label_purchase_90d, 0.0) AS label_purchase_90d,
  coalesce(labels_pivot.label_purchase_30d, 0.0) AS label_purchase_30d
FROM gold_user_features_asof AS features
LEFT JOIN labels_pivot
  ON features.user_id = labels_pivot.user_id
 AND features.as_of_date = labels_pivot.as_of_date
WHERE features.as_of_date = CAST('{as_of_date}' AS DATE)
ORDER BY features.user_id;
