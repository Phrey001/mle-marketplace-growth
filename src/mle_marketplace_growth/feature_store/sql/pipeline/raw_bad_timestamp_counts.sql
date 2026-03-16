-- Purpose: Count total raw rows and rows with invalid InvoiceDate timestamps.
-- Why: Shared-silver build fails fast if timestamp parse failures exceed the allowed threshold.
-- Select total rows and invalid timestamp count.
SELECT
  COUNT(*) AS total_rows,
  SUM(
    CASE
      WHEN TRY_STRPTIME(TRIM(CAST("InvoiceDate" AS VARCHAR)), '%Y-%m-%d %H:%M:%S') IS NULL
      THEN 1 ELSE 0
    END
  ) AS bad_timestamp_rows
FROM raw_source;
