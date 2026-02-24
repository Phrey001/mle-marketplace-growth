COPY (
  SELECT * FROM {table_name}
) TO '{output_path}' (HEADER, DELIMITER ',');
