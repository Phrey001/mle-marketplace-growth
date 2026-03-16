-- Purpose: Get max event_date from silver_transactions_line_items.
-- Why: Builders/debug helpers can use the latest available silver date as a runtime bound.
-- Select max event_date from silver.
SELECT MAX(event_date) FROM silver_transactions_line_items;
