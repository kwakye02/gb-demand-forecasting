"""Load, aggregate and resample historical GB weather data from Open-Meteo parquet chunks."""

import glob
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.utils.io import check_columns_match, get_parquet_columns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo" / "historical_weather" / "chunks"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

WEATHER_COLUMNS = ["time", "temperature_2m","relative_humidity_2m","precipitation",
                   "cloud_cover","wind_speed_10m","shortwave_radiation",]

def get_weather_files():
    """Collect all raw weather parquetfile paths."""
    return glob.glob(str(RAW_DATA_DIR / "*.parquet"))

def load_and_combine(files):
    """Validate schema consistency and concatenate all weather chunks."""
    check_columns_match(files, get_parquet_columns)
    dataframes = [pd.read_parquet(f) for f in files]
    return pd.concat(dataframes, ignore_index=True)

def select_columns(df, columns=WEATHER_COLUMNS):
    """Select the relevant subset of weather columns."""
    return df[columns].copy()

def aggregate_by_time(df):
    """Average weather variables across stations for each timestamp."""
    value_cols = df.drop(columns="time").columns.tolist()
    return df.groupby("time")[value_cols].mean().reset_index()

def resample_30min(df):
    """Resample to 30-minute intervals, forward-filling gaps."""
    return df.set_index("time").resample("30min").ffill().reset_index()

def run_historical_weather_pipeline(save=True):
    """Run the full weather data ingestion pipeline."""
    files = get_weather_files()

    combined_df = load_and_combine(files)
    combined_df = select_columns(combined_df)

    weather_gb = aggregate_by_time(combined_df)
    weather_gb = resample_30min(weather_gb)

    print(f"Final shape: {weather_gb.shape}")

    if save:
        out_path = PROCESSED_DATA_DIR / "historical_weather_all.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weather_gb.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    return weather_gb


if __name__ == "__main__":
    run_historical_weather_pipeline()
