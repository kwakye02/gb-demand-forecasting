"""Load and combine historical GB demand data (2009-2025) from NESO raw CSVs."""

import glob
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.utils.io import check_columns_match, find_common_columns, get_csv_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "neso"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_neso_dataset():
    """Collect all raw NESO demand CSV file paths."""
    return glob.glob(str(RAW_DATA_DIR / "*.csv"))

def extract_years(files):
    """Extract year from each filename, e.g. demanddata_2025.csv -> '2025'."""
    return [Path(f).stem.split("_")[-1] for f in files]

def group_by_schema_version(files, years):
    """Group files by known schema-change year ranges."""
    return {
        "2009_2018": [f for f, y in zip(files, years) if 2009 <= int(y) <= 2018],
        "2019_2022": [f for f, y in zip(files, years) if 2019 <= int(y) <= 2022],
        "2023_2025": [f for f, y in zip(files, years) if 2023 <= int(y)],
    }

def validate_schema_groups(groups):
    """Check that each schema group has internally consistent columns."""
    for label, group_files in groups.items():
        print(f"Checking group {label}...")
        check_columns_match(group_files, get_csv_columns)

def get_common_columns(sample_files):
    """Find columns common across a list of representative sample files."""
    schemas = [get_csv_columns(f) for f in sample_files]
    return find_common_columns(schemas, preserve_order =schemas[0])

def load_combined_demand_data(files, common_cols):
    """Load all years keeping only common columns, then concatenate."""
    dataframes = [pd.read_csv(f, usecols=common_cols) for f in files]
    return pd.concat(dataframes, ignore_index=True)

def run_neso_pipeline(save=True):
    """Run the full NESO demand data ingestion pipeline."""
    files = get_neso_dataset()
    years = extract_years(files)
    groups = group_by_schema_version(files, years)
    validate_schema_groups(groups)

    # use one representative file per schema group to find common columns
    sample_files = [groups["2009_2018"][0], groups["2019_2022"][0], groups["2023_2025"][0]]
    common_cols = get_common_columns(sample_files)

    combined_df = load_combined_demand_data(files, common_cols)

    print(f"Combined shape: {combined_df.shape}")
    print("Missing values:\n", combined_df.isnull().sum())

    if save:
        out_path = PROCESSED_DATA_DIR / "gb_demand_2009_2025.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    return combined_df


if __name__ == "__main__":
    run_neso_pipeline()
