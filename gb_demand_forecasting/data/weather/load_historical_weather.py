"""Load, aggregate and resample historical GB weather data from Open-Meteo parquet chunks."""

import glob
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.utils.io import check_columns_match, get_parquet_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_HISTORICAL_WEATHER_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "open_meteo" / "historical_weather" / "chunks"
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"

WEATHER_COLUMNS = [
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]


def get_historical_weather_files():
    """Collect all raw weather parquetfile paths."""
    return sorted(glob.glob(str(RAW_HISTORICAL_WEATHER_DIRECTORY / "*.parquet")))


def load_and_combine_weather_files(weather_files):
    """Validate schema consistency and concatenate all weather chunks."""
    check_columns_match(weather_files, get_parquet_columns)
    weather_frames = [pd.read_parquet(f) for f in weather_files]
    combined_weather = pd.concat(weather_frames, ignore_index=True)
    # Remove overlapping observations before averaging weather
    # across locations to avoid counting the same city twice.
    return combined_weather.drop_duplicates(["location_name", "time"], keep="last")


def select_weather_columns(weather_data, columns=WEATHER_COLUMNS):
    """Select the relevant subset of weather columns."""
    return weather_data[columns].copy()


def average_weather_across_locations(weather_data):
    """Average weather variables across stations for each timestamp."""
    weather_columns = weather_data.drop(columns="time").columns.tolist()
    return weather_data.groupby("time")[weather_columns].mean().reset_index()


def expand_hourly_to_half_hourly(weather_data):
    """
    Convert hourly weather observations to half-hourly records.

    Repeat each hourly value at the following half-hour.
    Preserve missing hours rather than filling them with
    observations from earlier periods.
    """
    weather_data = weather_data.set_index("time")
    half_hour_timestamps = pd.date_range(
        weather_data.index.min(), weather_data.index.max() + pd.Timedelta(minutes=30), freq="30min", name="time"
    )
    half_hourly_weather = weather_data.reindex(half_hour_timestamps)
    is_half_hour = half_hour_timestamps.minute == 30
    half_hourly_weather[is_half_hour] = half_hourly_weather.shift(1)[is_half_hour]
    return half_hourly_weather.reset_index()


def prepare_historical_weather(save=True):
    """Run the full weather data ingestion pipeline."""
    weather_files = get_historical_weather_files()

    combined_weather = load_and_combine_weather_files(weather_files)
    combined_weather = select_weather_columns(combined_weather)

    gb_weather = average_weather_across_locations(combined_weather)
    gb_weather = expand_hourly_to_half_hourly(gb_weather)

    print(f"Final shape: {gb_weather.shape}")

    if save:
        output_path = PROCESSED_DATA_DIRECTORY / "historical_weather_all.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gb_weather.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

    return gb_weather


if __name__ == "__main__":
    prepare_historical_weather()
