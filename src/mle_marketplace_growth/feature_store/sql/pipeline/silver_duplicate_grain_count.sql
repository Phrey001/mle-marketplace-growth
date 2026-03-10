-- Purpose: Count duplicate (invoice_id, item_id, event_ts) rows in silver_transactions_line_items.
-- Select count of duplicate silver grain rows.
SELECT COUNT(*)
FROM (
  -- Select invoice/item/timestamp groups with more than one row.
  SELECT invoice_id, item_id, event_ts
  FROM silver_transactions_line_items
  GROUP BY invoice_id, item_id, event_ts
  HAVING COUNT(*) > 1
) AS duplicates;
