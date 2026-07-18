import argparse
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data import download_resource, get_demand_resources


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "neso"

DEFAULT_YEAR = 2025
REQUIRED_COLUMNS = {
    "SETTLEMENT_DATE",
    "SETTLEMENT_PERIOD",
    "ND",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download one or more years of NESO Historic Demand Data."
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        help=f"Single historic demand year to download. Defaults to {DEFAULT_YEAR}.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="First historic demand year to download.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last historic demand year to download.",
    )

    args = parser.parse_args()

    if args.year is not None and (
        args.start_year is not None or args.end_year is not None
    ):
        parser.error("Use either a single year or --start-year/--end-year, not both.")

    if (args.start_year is None) != (args.end_year is None):
        parser.error("--start-year and --end-year must be provided together.")

    if (
        args.start_year is not None
        and args.end_year is not None
        and args.start_year > args.end_year
    ):
        parser.error("--start-year cannot be greater than --end-year.")

    if args.year is None and args.start_year is None:
        args.year = DEFAULT_YEAR

    return args


def validate_dataset(file_path):
    dataframe = pd.read_csv(file_path)

    missing_columns = (
        REQUIRED_COLUMNS - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    dates = pd.to_datetime(
        dataframe["SETTLEMENT_DATE"],
        errors="coerce",
    )

    duplicates = dataframe.duplicated(
        ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
    ).sum()

    print("\nValidation summary")
    print("Shape:", dataframe.shape)
    print("Date range:", dates.min(), "to", dates.max())
    print("Invalid dates:", dates.isna().sum())
    print("Missing ND:", dataframe["ND"].isna().sum())
    print("Duplicate observations:", duplicates)

    return dataframe


def main(
    year=None,
    start_year=None,
    end_year=None,
    output_directory=RAW_DATA_DIRECTORY,
):
    if year is not None and (
        start_year is not None or end_year is not None
    ):
        raise ValueError("Use either year or start_year/end_year, not both.")

    if year is not None:
        start_year = year
        end_year = year
    elif start_year is None and end_year is None:
        start_year = DEFAULT_YEAR
        end_year = DEFAULT_YEAR
    elif start_year is None or end_year is None:
        raise ValueError("start_year and end_year must be provided together.")

    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year.")

    resources = get_demand_resources(
        start_year=start_year,
        end_year=end_year,
    )

    file_paths = []

    for resource in resources:
        file_path = download_resource(
            resource,
            output_directory,
        )

        validate_dataset(file_path)
        file_paths.append(file_path)

    return file_paths


def interactive_prompt():
    """Prompt the user interactively for year selection."""
    print("NESO Historic Demand Data Downloader")
    print("=====================================")
    mode = input("Download a single year or a range? [single/range]: ").strip().lower()

    if mode == "range":
        start_year = int(input("Enter start year: ").strip())
        end_year = int(input("Enter end year: ").strip())
        main(start_year=start_year, end_year=end_year)
    else:
        year = input(f"Enter year (default {DEFAULT_YEAR}): ").strip()
        main(year=int(year) if year else DEFAULT_YEAR)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        args = parse_args()
        if args.start_year is not None:
            main(start_year=args.start_year, end_year=args.end_year)
        else:
            main(year=args.year)
    else:
        interactive_prompt()
