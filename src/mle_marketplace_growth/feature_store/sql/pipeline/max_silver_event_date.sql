-- Purpose: Get max event_date from silver_transactions_line_items.
-- Select max event_date from silver.
SELECT MAX(event_date) FROM silver_transactions_line_items;
