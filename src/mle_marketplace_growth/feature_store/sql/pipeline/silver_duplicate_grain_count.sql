SELECT count(*)
FROM (
  SELECT invoice_id, item_id, event_ts
  FROM silver_transactions_line_items
  GROUP BY invoice_id, item_id, event_ts
  HAVING count(*) > 1
) AS duplicates;
