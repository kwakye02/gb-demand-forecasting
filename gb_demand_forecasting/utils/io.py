"""Shared I/O utility functions for validation across raw data files."""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def get_csv_columns(file_path):
    """Read only the header row of a CSV file to get its columns."""
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    return columns

def get_parquet_columns(file_path):
    """Get column names of a parquet file without loading data."""
    columns = pq.ParquetFile(file_path).schema_arrow.names
    return columns


def check_columns_match(file_paths, reader):
    """Check if all files share identical columns.
    Args:
        file_paths: list of file paths to check.
        reader: function takes a file path and returns list of column names.
    """
    if not file_paths:
        raise ValueError("No files.")

    reference_cols = reader(file_paths[0])
    for f in file_paths[1:]:
        cols = reader(f)
        assert reference_cols == cols, f"Column mismatch in {Path(f).name}"

    print(f"All {len(file_paths)} files have identical columns.")


def find_common_columns(columns_list, preserve_order=None):
    """Find columns common to all schemas.
    Args:
        columns_list: list of column-name lists.
        preserve_order: optional reference schema to preserve column order.
    """
    common = set(columns_list[0])
    for columns in columns_list[1:]:
        common &= set(columns)

    if preserve_order:
        return [col for col in preserve_order if col in common]
    return list(common)
