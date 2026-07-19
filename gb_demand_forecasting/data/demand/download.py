import argparse
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "neso"

METADATA_URL = "https://api.neso.energy/api/3/action/package_show"
DATASET_ID = "historic-demand-data"

DEFAULT_YEAR = 2025
REQUIRED_COLUMNS = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"}


def get_demand_resources(start_year=2009, end_year=2025):
    """Return NESO demand resources within the selected period."""

    response = requests.get(METADATA_URL, params={"id": DATASET_ID}, timeout=30)
    response.raise_for_status()

    metadata = response.json()

    if not metadata.get("success"):
        raise RuntimeError("NESO metadata request failed.")

    selected_resources = []

    for resource in metadata["result"]["resources"]:
        name = resource.get("name", "")
        year_text = name.split()[-1]

        if resource.get("format") != "CSV":
            continue

        if not year_text.isdigit():
            continue

        year = int(year_text)

        if start_year <= year <= end_year:
            selected_resources.append(
                {
                    "year": year,
                    "id": resource["id"],
                    "url": resource["url"],
                }
            )

    selected_resources.sort(key=lambda item: item["year"])

    expected_years = set(range(start_year, end_year + 1))
    available_years = {
        item["year"] for item in selected_resources
    }

    if available_years != expected_years:
        missing = expected_years - available_years
        raise RuntimeError(f"Missing years: {sorted(missing)}")

    return selected_resources


def download_resource(resource, output_directory):
    """Download one NESO CSV unless it already exists."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    file_path = (
        output_directory
        / f"demanddata_{resource['year']}.csv"
    )

    if file_path.exists() and file_path.stat().st_size > 0:
        print(f"{file_path.name} already exists.")
        return file_path

    response = requests.get(
        resource["url"],
        timeout=120,
    )
    response.raise_for_status()

    file_path.write_bytes(response.content)

    print(f"Downloaded {file_path.name}")
    return file_path


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
