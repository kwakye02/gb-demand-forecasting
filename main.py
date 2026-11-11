"""Download and build all data up to a chosen end date.

Run from the project root:

    python main.py                         # refresh everything up to yesterday and rebuild
    python main.py --end-date 2026-10-01   # stop at a fixed date
    python main.py --skip-download         # only rebuild the processed files
    python main.py --live                  # also save a live weather forecast snapshot

Steps
1. Elexon national demand (INDO) and every published day-ahead forecast version (NDF),
   one file per month. The month containing the end date is always downloaded again.
2. Open-Meteo actual weather (ERA5), yearly chunks. ERA5 runs about 6 days behind, so the
   latest days are empty and are dropped when the dataset is built.
3. Open-Meteo 48-hour-ahead forecasts (previous runs, one pinned model), monthly chunks.
   Chunks already on disk are reused.
4. NESO embedded wind and solar forecast: for each half-hour, the newest version issued by
   08:30 UK on the day before (selected on NESO's query API; one small file per month).
5. Rebuild data/processed/: demand, weather, forecast weather, embedded forecast, and the
   merged dataset (data/processed/gb_demand_merged.parquet) that the notebooks start from.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data.build_dataset import build_dataset
from gb_demand_forecasting.data.demand import elexon_demand
from gb_demand_forecasting.data.demand.load_elexon import run_elexon_pipeline
from gb_demand_forecasting.data.embedded_forecast import (
    build_embedded_forecast_file,
    download_embedded_forecasts,
)
from gb_demand_forecasting.data.weather.api import (
    PINNED_PREVIOUS_RUNS_MODEL,
    build_chunk_output_path,
    create_request_session,
    download_live_forecast,
    fetch_chunked_historical_weather,
    fetch_chunked_previous_runs,
    previous_runs_chunk_directory,
)
from gb_demand_forecasting.data.weather.load_historical_weather import (
    run_historical_weather_pipeline,
)
from gb_demand_forecasting.data.weather.load_weather_previous_runs import (
    run_previous_runs_pipeline,
)

DEMAND_START_DATE = "2020-01-01"
WEATHER_START_DATE = "2020-01-01"
# 48-hour-ahead forecasts, one pinned model, from 2024-07-01. Jul-Dec 2024 is
# bias-correction history only (training stays on actual weather).
PREVIOUS_RUNS_START_DATE = "2024-07-01"
PREVIOUS_RUNS_LEAD_DAYS = (2,)
PREVIOUS_RUNS_SLEEP_SECONDS = 10


def chunk_directory(source):
    return build_chunk_output_path(source, "start", "end").parent


def prune_superseded_chunks(directory):
    """Delete a partial chunk once a longer chunk with the same start date is on disk."""
    longest = {}

    for file in Path(directory).glob("gb_*_*_*.parquet"):
        *_, start, end = file.stem.split("_")
        if start not in longest or end > longest[start][0]:
            longest[start] = (end, file)

    for file in Path(directory).glob("gb_*_*_*.parquet"):
        *_, start, end = file.stem.split("_")
        if end < longest[start][0]:
            print(f"Removing superseded chunk {file.name}")
            file.unlink()


def download_demand(end_date):
    end = end_date.strftime("%Y-%m-%d")
    print(f"\n=== Elexon national demand and day-ahead forecasts to {end} ===")
    elexon_demand.download_national_demand(DEMAND_START_DATE, end)
    elexon_demand.download_day_ahead_forecasts(DEMAND_START_DATE, end)


def download_embedded(end_date):
    end = end_date.strftime("%Y-%m-%d")
    print(f"\n=== NESO embedded wind and solar forecast to {end} ===")
    download_embedded_forecasts(DEMAND_START_DATE, end)


def download_weather(end_date):
    end = end_date.strftime("%Y-%m-%d")
    print(f"\n=== Open-Meteo actual weather to {end} ===")
    fetch_chunked_historical_weather(
        WEATHER_START_DATE,
        end,
        session=create_request_session(expire_after=None),
        chunk_cache_directory=chunk_directory("historical_weather"),
    )
    prune_superseded_chunks(chunk_directory("historical_weather"))


def download_previous_runs(end_date):
    end = end_date.strftime("%Y-%m-%d")
    print(f"\n=== Open-Meteo 48h-ahead forecasts ({PINNED_PREVIOUS_RUNS_MODEL}) to {end} ===")
    directory = previous_runs_chunk_directory(PINNED_PREVIOUS_RUNS_MODEL)
    fetch_chunked_previous_runs(
        PREVIOUS_RUNS_START_DATE,
        end,
        lead_days=PREVIOUS_RUNS_LEAD_DAYS,
        session=create_request_session(expire_after=None),
        chunk_cache_directory=directory,
        sleep_seconds=PREVIOUS_RUNS_SLEEP_SECONDS,
        models=PINNED_PREVIOUS_RUNS_MODEL,
    )
    prune_superseded_chunks(directory)


def process(end_date):
    print("\n=== Processing ===")
    run_elexon_pipeline()
    run_historical_weather_pipeline()
    run_previous_runs_pipeline(lead_day=PREVIOUS_RUNS_LEAD_DAYS[0])
    build_embedded_forecast_file()
    build_dataset(end_date=end_date)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last day to download and keep (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads and only rebuild the processed files.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also save a timestamped live weather forecast snapshot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    end_date = pd.Timestamp(args.end_date or date.today() - timedelta(days=1))

    if not args.skip_download:
        download_demand(end_date)
        download_embedded(end_date)
        download_weather(end_date)
        download_previous_runs(end_date)

    if args.live:
        print("\n=== Live weather forecast snapshot ===")
        download_live_forecast()

    process(end_date)


if __name__ == "__main__":
    main()
