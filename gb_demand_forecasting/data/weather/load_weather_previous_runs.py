"""Load, aggregate and resample GB previous-runs weather forecasts from Open-Meteo parquet chunks."""

import glob
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data.weather.api import (
    PINNED_PREVIOUS_RUNS_MODEL,
    previous_runs_chunk_directory,
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


def get_weather_files(model=PINNED_PREVIOUS_RUNS_MODEL):
    """Collect the raw previous-runs parquet files for one pinned model."""
    return glob.glob(str(previous_runs_chunk_directory(model) / "*.parquet"))


def load_and_combine(files):
    """Validate schema consistency and concatenate all weather chunks."""
    check_columns_match(files, get_parquet_columns)
    dataframes = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dataframes, ignore_index=True)
    return combined.drop_duplicates(["location_name", "time"], keep="last")


def select_lead_day(df, lead_day=DEFAULT_LEAD_DAY):
    """Select one lead time and rename its columns to the plain variable names."""
    df = df.copy()
    renamed = {f"{variable}_previous_day{lead_day}": variable for variable in WEATHER_VARIABLES}
    missing = set(renamed).difference(df.columns)

    if missing:
        raise ValueError(f"Lead day {lead_day} not in the raw data: {sorted(missing)}")

    return df[["time", *renamed]].rename(columns=renamed)


def aggregate_by_time(df):
    """Average weather variables across stations for each timestamp."""
    value_cols = df.drop(columns="time").columns.tolist()
    return df.groupby("time")[value_cols].mean().reset_index()


def resample_30min(df):
    """Convert hourly values to half-hourly by repeating each hour into its :30 slot.

    A missing hour stays missing in both of its half-hours: nothing is carried across
    gaps. The index is extended by one step so the last hour keeps its :30 row.
    """
    df = df.set_index("time")
    index = pd.date_range(
        df.index.min(), df.index.max() + pd.Timedelta(minutes=30), freq="30min", name="time"
    )
    half_hourly = df.reindex(index)
    on_half_hour = index.minute == 30
    half_hourly[on_half_hour] = half_hourly.shift(1)[on_half_hour]
    return half_hourly.reset_index()


def run_previous_runs_pipeline(
    lead_day=DEFAULT_LEAD_DAY, save=True, model=PINNED_PREVIOUS_RUNS_MODEL
):
    """Run the full previous-runs weather ingestion pipeline for one lead time."""
    files = get_weather_files(model)

    combined_df = load_and_combine(files)
    combined_df = select_lead_day(combined_df, lead_day)

    weather_gb = aggregate_by_time(combined_df)
    weather_gb = resample_30min(weather_gb)

    print(f"Final shape: {weather_gb.shape}")

    if save:
        out_path = PROCESSED_DATA_DIR / f"weather_previous_runs_day{lead_day}_{model}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weather_gb.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    return weather_gb


if __name__ == "__main__":
    run_previous_runs_pipeline()
