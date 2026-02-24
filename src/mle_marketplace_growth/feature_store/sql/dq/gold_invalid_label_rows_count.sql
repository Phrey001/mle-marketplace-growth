SELECT COUNT(*)
FROM gold_labels
WHERE label_name NOT IN ('net_revenue_30d', 'purchase_30d')
   OR window_days <> 30;
