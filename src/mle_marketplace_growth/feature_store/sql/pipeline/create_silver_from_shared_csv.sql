CREATE OR REPLACE TEMP TABLE silver_transactions_line_items AS
SELECT
  trim(CAST(invoice_id AS VARCHAR)) AS invoice_id,
  trim(CAST(item_id AS VARCHAR)) AS item_id,
  nullif(trim(CAST(item_description AS VARCHAR)), '') AS item_description,
  CAST(try_cast(trim(CAST(quantity AS VARCHAR)) AS DOUBLE) AS INTEGER) AS quantity,
  try_strptime(trim(CAST(event_ts AS VARCHAR)), '%Y-%m-%d %H:%M:%S') AS event_ts,
  try_cast(trim(CAST(event_date AS VARCHAR)) AS DATE) AS event_date,
  try_cast(trim(CAST(unit_price AS VARCHAR)) AS DOUBLE) AS unit_price,
  try_cast(trim(CAST(line_revenue AS VARCHAR)) AS DOUBLE) AS line_revenue,
  trim(CAST(user_id AS VARCHAR)) AS user_id,
  trim(CAST(country AS VARCHAR)) AS country
FROM read_csv_auto('{silver_csv}', header=TRUE, all_varchar=TRUE, ignore_errors=FALSE);
