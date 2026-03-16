-- Purpose: Create temp view raw_source from input CSV.
-- Why: Later SQL can read the raw source through one stable relation name.
CREATE OR REPLACE TEMP VIEW raw_source AS
-- Select all raw CSV rows into a temp view.
SELECT *
FROM read_csv_auto('{input_csv}', header=true, sample_size=-1);
