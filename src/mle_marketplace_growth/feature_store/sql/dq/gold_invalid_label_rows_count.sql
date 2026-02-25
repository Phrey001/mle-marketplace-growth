SELECT COUNT(*)
FROM gold_labels
WHERE (label_name = 'net_revenue_30d' AND window_days <> 30)
   OR (label_name = 'purchase_30d' AND window_days <> 30)
   OR (label_name = 'net_revenue_60d' AND window_days <> 60)
   OR (label_name = 'purchase_60d' AND window_days <> 60)
   OR (label_name = 'net_revenue_90d' AND window_days <> 90)
   OR (label_name = 'purchase_90d' AND window_days <> 90)
   OR label_name NOT IN (
       'net_revenue_30d', 'purchase_30d',
       'net_revenue_60d', 'purchase_60d',
       'net_revenue_90d', 'purchase_90d'
   );
