CREATE OR REPLACE TABLE silver_transactions_line_items AS
WITH raw_deduplicated AS (
  SELECT DISTINCT
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
  FROM raw_source
),
typed_source AS (
  SELECT
    trim(CAST("Invoice" AS VARCHAR)) AS invoice_id,
    trim(CAST("StockCode" AS VARCHAR)) AS item_id,
    nullif(trim(CAST("Description" AS VARCHAR)), '') AS item_description_raw,
    try_cast(trim(CAST("Quantity" AS VARCHAR)) AS DOUBLE) AS quantity_raw,
    try_strptime(trim(CAST("InvoiceDate" AS VARCHAR)), '%Y-%m-%d %H:%M:%S') AS event_ts,
    try_cast(trim(CAST("Price" AS VARCHAR)) AS DOUBLE) AS unit_price_raw,
    trim(CAST("Customer ID" AS VARCHAR)) AS user_id_raw,
    trim(CAST("Country" AS VARCHAR)) AS country
  FROM raw_deduplicated
)
SELECT
  invoice_id,
  item_id,
  item_description_raw AS item_description,
  cast(quantity_raw AS INTEGER) AS quantity,
  event_ts,
  cast(event_ts AS DATE) AS event_date,
  unit_price_raw AS unit_price,
  round(quantity_raw * unit_price_raw, 4) AS line_revenue,
  CASE
    WHEN user_id_raw LIKE '%.0'
         AND regexp_full_match(substr(user_id_raw, 1, length(user_id_raw) - 2), '^[0-9]+$')
      THEN substr(user_id_raw, 1, length(user_id_raw) - 2)
    ELSE user_id_raw
  END AS user_id,
  country
FROM typed_source
WHERE invoice_id <> ''
  AND item_id <> ''
  AND event_ts IS NOT NULL
  AND quantity_raw IS NOT NULL
  AND quantity_raw <> 0
  AND unit_price_raw IS NOT NULL
  AND unit_price_raw >= 0;
