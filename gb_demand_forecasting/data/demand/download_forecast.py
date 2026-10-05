import argparse
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Separate folder: load_neso globs every *.csv in data/raw/neso.
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "neso_forecast"
OUTPUT_FILENAME = "day_ahead_demand_forecast.csv"

METADATA_URL = "https://api.neso.energy/api/3/action/package_show"
DATASET_ID = "day-ahead-half-hourly-demand-forecast-performance"

REQUIRED_COLUMNS = {
    "Date",
    "Datetime",
    "Settlement_Period",
    "Demand_Forecast",
    "Demand_Outturn",
    "Publish_Datetime",
}


def get_forecast_resource():
    """Return the NESO day-ahead forecast performance CSV resource."""

    response = requests.get(METADATA_URL, params={"id": DATASET_ID}, timeout=30)
    response.raise_for_status()

    metadata = response.json()

    if not metadata.get("success"):
        raise RuntimeError("NESO metadata request failed.")

    csv_resources = [
        resource
        for resource in metadata["result"]["resources"]
        if resource.get("format") == "CSV"
    ]

    if len(csv_resources) != 1:
        raise RuntimeError(
            f"Expected one CSV resource, found {len(csv_resources)}."
        )

    return csv_resources[0]


def download_forecast(output_directory=RAW_DATA_DIRECTORY, refresh=False):
    """Download the forecast CSV unless it already exists.

    The NESO file is updated as new days are published, so use refresh=True
    to replace an existing copy.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    file_path = output_directory / OUTPUT_FILENAME

    if not refresh and file_path.exists() and file_path.stat().st_size > 0:
        print(f"{file_path.name} already exists.")
        return file_path

    resource = get_forecast_resource()

    response = requests.get(resource["url"], timeout=120)
    response.raise_for_status()

    file_path.write_bytes(response.content)

    print(f"Downloaded {file_path.name}")
    return file_path


def validate_dataset(file_path):
    dataframe = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise RuntimeError(f"Missing columns: {sorted(missing_columns)}")

    datetimes = pd.to_datetime(dataframe["Datetime"], errors="coerce")

    duplicates = dataframe.duplicated(["Date", "Settlement_Period"]).sum()

    print("\nValidation summary")
    print("Shape:", dataframe.shape)
    print("Date range:", datetimes.min(), "to", datetimes.max())
    print("Invalid datetimes:", datetimes.isna().sum())
    print("Missing Demand_Forecast:", dataframe["Demand_Forecast"].isna().sum())
    print("Duplicate observations:", duplicates)

    return dataframe


def main(output_directory=RAW_DATA_DIRECTORY, refresh=False):
    file_path = download_forecast(output_directory, refresh=refresh)
    validate_dataset(file_path)
    return file_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NESO Day Ahead Half Hourly Demand Forecast Performance."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(refresh=args.refresh)
