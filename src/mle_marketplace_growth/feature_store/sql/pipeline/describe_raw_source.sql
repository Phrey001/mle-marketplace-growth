-- Purpose: Describe raw_source schema.
-- Why: Shared-silver build validates required raw columns before any transformation runs.
-- Select raw_source for schema inspection.
DESCRIBE SELECT * FROM raw_source;
