SELECT
  COUNT(*) AS total_rows,
  SUM(
    CASE
      WHEN try_strptime(trim(CAST("InvoiceDate" AS VARCHAR)), '%Y-%m-%d %H:%M:%S') IS NULL
      THEN 1 ELSE 0
    END
  ) AS bad_timestamp_rows
FROM raw_source;
