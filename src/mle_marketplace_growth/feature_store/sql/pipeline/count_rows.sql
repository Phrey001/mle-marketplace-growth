-- Purpose: Count rows in {table_name}.
-- Why: Builders use the count in run manifests after materializing each table.
-- Select row count for {table_name}.
SELECT COUNT(*) FROM {table_name};
