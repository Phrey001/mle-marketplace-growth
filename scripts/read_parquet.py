"""Convert a parquet file to CSV using DuckDB."""

from __future__ import annotations
import argparse
from pathlib import Path

import duckdb


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {resolved}")
    return resolved


def _default_output_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a parquet file to CSV.")
    parser.add_argument("input_parquet", help="Input parquet file path")
    args = parser.parse_args()

    parquet_path = _resolve_path(args.input_parquet)
    output_path = _default_output_path(parquet_path)
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_progress_bar=false")

    query = f"SELECT * FROM read_parquet('{parquet_path}')"
    con.execute(f"COPY ({query}) TO '{output_path}' (HEADER, DELIMITER ',');")
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
