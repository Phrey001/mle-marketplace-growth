CREATE OR REPLACE TEMP VIEW raw_source AS
SELECT *
FROM read_csv_auto('{input_csv}', header=true, sample_size=-1);
