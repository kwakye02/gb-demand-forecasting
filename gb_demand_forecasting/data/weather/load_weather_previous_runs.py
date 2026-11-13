"""
Prepare archived Open-Meteo weather forecasts for demand modelling.

Selects a fixed forecast lead time, combines weather across
nine GB locations, and converts hourly forecasts into
half-hourly records.
"""

import glob
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data.weather.api import (
    PINNED_PREVIOUS_RUNS_MODEL,
    get_previous_runs_chunk_directory,
)
from gb_demand_forecasting.utils.io import check_columns_match, get_parquet_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]

DEFAULT_LEAD_DAY = 2


def get_previous_run_files(weather_model=PINNED_PREVIOUS_RUNS_MODEL):
    """Collect the raw previous-runs parquet files for one pinned model."""
    return sorted(glob.glob(str(get_previous_runs_chunk_directory(weather_model) / "*.parquet")))


def load_and_combine_weather_files(weather_files):
    """Validate schema consistency and concatenate all weather chunks."""
    check_columns_match(weather_files, get_parquet_columns)
    weather_frames = [pd.read_parquet(f) for f in weather_files]
    combined_weather = pd.concat(weather_frames, ignore_index=True)
    return combined_weather.drop_duplicates(["location_name", "time"], keep="last")


def select_forecast_lead_day(weather_data, lead_day=DEFAULT_LEAD_DAY):
    """Select one lead time and rename its columns to the plain variable names."""
    weather_data = weather_data.copy()
    column_name_map = {
        f"{variable}_previous_day{lead_day}": variable for variable in WEATHER_VARIABLES
    }
    missing_columns = set(column_name_map).difference(weather_data.columns)

    if missing_columns:
        raise ValueError(f"Lead day {lead_day} not in the raw data: {sorted(missing_columns)}")

    return weather_data[["time", *column_name_map]].rename(columns=column_name_map)


def average_weather_across_locations(weather_data):
    """Average weather variables across stations for each timestamp."""
    weather_columns = weather_data.drop(columns="time").columns.tolist()
    return weather_data.groupby("time")[weather_columns].mean().reset_index()


def expand_hourly_to_half_hourly(weather_data):
    """Convert hourly values to half-hourly by repeating each hour into its :30 slot.

    A missing hour stays missing in both of its half-hours: nothing is carried across
    gaps. The index is extended by one step so the last hour keeps its :30 row.
    """
    weather_data = weather_data.set_index("time")
    half_hour_timestamps = pd.date_range(
        weather_data.index.min(),
        weather_data.index.max() + pd.Timedelta(minutes=30),
        freq="30min",
        name="time",
    )
    half_hourly_weather = weather_data.reindex(half_hour_timestamps)
    is_half_hour = half_hour_timestamps.minute == 30
    half_hourly_weather[is_half_hour] = half_hourly_weather.shift(1)[is_half_hour]
    return half_hourly_weather.reset_index()


def prepare_previous_run_weather(
    lead_day=DEFAULT_LEAD_DAY, save=True, weather_model=PINNED_PREVIOUS_RUNS_MODEL
):
    """Run the full previous-runs weather ingestion pipeline for one lead time."""
    weather_files = get_previous_run_files(weather_model)

    combined_weather = load_and_combine_weather_files(weather_files)
    combined_weather = select_forecast_lead_day(combined_weather, lead_day)

    gb_weather = average_weather_across_locations(combined_weather)
    gb_weather = expand_hourly_to_half_hourly(gb_weather)

    print(f"Final shape: {gb_weather.shape}")

    if save:
        output_path = (
            PROCESSED_DATA_DIR / f"weather_previous_runs_day{lead_day}_{weather_model}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gb_weather.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

    return gb_weather


if __name__ == "__main__":
    prepare_previous_run_weather()
