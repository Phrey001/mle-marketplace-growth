WITH normalized AS (
  SELECT
    coalesce(trim(CAST("Invoice" AS VARCHAR)), '') AS invoice,
    coalesce(trim(CAST("StockCode" AS VARCHAR)), '') AS stock_code,
    coalesce(trim(CAST("Description" AS VARCHAR)), '') AS description,
    coalesce(trim(CAST("Quantity" AS VARCHAR)), '') AS quantity,
    coalesce(trim(CAST("InvoiceDate" AS VARCHAR)), '') AS invoice_date,
    coalesce(trim(CAST("Price" AS VARCHAR)), '') AS price,
    coalesce(trim(CAST("Customer ID" AS VARCHAR)), '') AS customer_id,
    coalesce(trim(CAST("Country" AS VARCHAR)), '') AS country
  FROM raw_source
)
SELECT
  COUNT(*) - COUNT(
    DISTINCT md5(
      invoice || '|' || stock_code || '|' || description || '|' || quantity || '|' ||
      invoice_date || '|' || price || '|' || customer_id || '|' || country
    )
  ) AS exact_duplicate_rows
FROM normalized;
