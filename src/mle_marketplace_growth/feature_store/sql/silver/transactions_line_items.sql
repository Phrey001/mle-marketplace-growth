-- Purpose: Build canonical silver_transactions_line_items from raw_source.
CREATE OR REPLACE TABLE silver_transactions_line_items AS
WITH
raw_deduplicated AS (
  -- Select distinct raw rows at the source grain.
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
  -- Select typed and normalized raw columns.
  SELECT
    TRIM(CAST("Invoice" AS VARCHAR)) AS invoice_id,
    TRIM(CAST("StockCode" AS VARCHAR)) AS item_id,
    NULLIF(TRIM(CAST("Description" AS VARCHAR)), '') AS item_description_raw,
    TRY_CAST(TRIM(CAST("Quantity" AS VARCHAR)) AS DOUBLE) AS quantity_raw,
    TRY_STRPTIME(TRIM(CAST("InvoiceDate" AS VARCHAR)), '%Y-%m-%d %H:%M:%S') AS event_ts,
    TRY_CAST(TRIM(CAST("Price" AS VARCHAR)) AS DOUBLE) AS unit_price_raw,
    TRIM(CAST("Customer ID" AS VARCHAR)) AS user_id_raw,
    TRIM(CAST("Country" AS VARCHAR)) AS country
  FROM raw_deduplicated
),
valid_source AS (
  -- Select valid rows and normalize user_id formatting.
  SELECT
    invoice_id,
    item_id,
    item_description_raw,
    quantity_raw,
    event_ts,
    unit_price_raw,
    CASE
      -- Normalize numeric-looking IDs like "12345.0" to "12345".
      -- Regex requires the prefix to be digits only (^[0-9]+$) before stripping ".0".
      -- ^ and $ anchor the whole string; [0-9]+ means one or more digits.
      WHEN user_id_raw LIKE '%.0'
           AND REGEXP_FULL_MATCH(SUBSTR(user_id_raw, 1, LENGTH(user_id_raw) - 2), '^[0-9]+$')
        THEN SUBSTR(user_id_raw, 1, LENGTH(user_id_raw) - 2)
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
    AND unit_price_raw >= 0
),
grain_aggregated AS (
  -- Aggregate to the canonical silver grain (invoice_id, item_id, event_ts).
  -- This collapses any duplicated PK blocks that remain after raw de-duplication.
  SELECT
    invoice_id,
    item_id,
    event_ts,
    MAX(item_description_raw) AS item_description,
    CAST(SUM(quantity_raw) AS INTEGER) AS quantity,
    ROUND(AVG(unit_price_raw), 4) AS unit_price,
    -- Use line_revenue = sum(quantity * unit_price) at the aggregated grain.
    ROUND(SUM(quantity_raw * unit_price_raw), 4) AS line_revenue,
    MAX(NULLIF(user_id, '')) AS user_id,
    MAX(NULLIF(country, '')) AS country
  FROM valid_source
  GROUP BY invoice_id, item_id, event_ts
)
-- Select final canonical silver rows.
SELECT
  invoice_id,
  item_id,
  item_description,
  quantity,
  event_ts,
  CAST(event_ts AS DATE) AS event_date,
  unit_price,
  line_revenue,
  user_id,
  country
FROM grain_aggregated
WHERE quantity <> 0;
