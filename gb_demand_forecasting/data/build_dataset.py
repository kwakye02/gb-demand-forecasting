"""Merge Elexon demand, actual weather, 48-hour forecast weather and the day-ahead benchmark.
It writes data/processed/gb_demand_merged.parquet, indexed by UTC datetime.
"""

from pathlib import Path

import holidays
import pandas as pd

from gb_demand_forecasting.data.demand.elexon_demand import (
    BENCHMARK_CUTOFF_TIME,
    PREDICTION_TIME,
    build_day_ahead_benchmark,
)
from gb_demand_forecasting.data.weather.api import PINNED_PREVIOUS_RUNS_MODEL
from gb_demand_forecasting.data.weather.load_weather_previous_runs import DEFAULT_LEAD_DAY

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEMAND_PATH = PROCESSED_DATA_DIR / "gb_demand_elexon.parquet"
WEATHER_PATH = PROCESSED_DATA_DIR / "historical_weather_all.parquet"
WEATHER_FORECAST_PATH = (
    PROCESSED_DATA_DIR
    / f"weather_previous_runs_day{DEFAULT_LEAD_DAY}_{PINNED_PREVIOUS_RUNS_MODEL}.parquet"
)
EMBEDDED_PATH = PROCESSED_DATA_DIR / "embedded_forecast_0830.parquet"
OUTPUT_PATH = PROCESSED_DATA_DIR / "gb_demand_merged.parquet"

KEY = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]


def add_settlement_timestamps(demand):
    """Build UK-local and UTC timestamps for every settlement period."""
    demand = demand.copy()
    demand["SETTLEMENT_DATE"] = pd.to_datetime(demand["SETTLEMENT_DATE"], format="mixed")

    # Settlement day starts at UK local midnight; each period is 30 minutes.
    settlement_midnight = demand["SETTLEMENT_DATE"].dt.tz_localize("Europe/London")
    demand["datetime_local"] = settlement_midnight + pd.to_timedelta(
        (demand["SETTLEMENT_PERIOD"] - 1) * 30, unit="minutes"
    )
    demand["datetime"] = demand["datetime_local"].dt.tz_convert("UTC")
    demand["local_date"] = demand["SETTLEMENT_DATE"].dt.normalize()
    demand["local_time"] = demand["datetime_local"].dt.strftime("%H:%M")

    if demand["datetime"].duplicated().any():
        raise ValueError("Duplicate demand timestamps.")

    return demand


def add_benchmark(demand, benchmark):
    """Attach the newest day-ahead forecast published by the benchmark cutoff."""
    benchmark = benchmark.rename(
        columns={
            "settlement_date": "SETTLEMENT_DATE",
            "settlement_period": "SETTLEMENT_PERIOD",
        }
    )
    return demand.merge(benchmark, on=KEY, how="left", validate="one_to_one")


def add_embedded_forecast(demand, embedded):
    """Attach NESO's embedded wind and solar forecast (newest issued by 08:30 UK on D-1)."""
    embedded = embedded.assign(SETTLEMENT_DATE=pd.to_datetime(embedded["SETTLEMENT_DATE"]))
    return demand.merge(embedded, on=KEY, how="left", validate="one_to_one")


def add_weather(demand, weather, weather_forecast):
    """Attach actual weather, and the 48-hour forecast weather with a forecast_ prefix."""
    weather = weather.assign(datetime=pd.to_datetime(weather["time"])).drop(columns="time")
    weather_forecast = weather_forecast.add_prefix("forecast_")
    weather_forecast = weather_forecast.rename(columns={"forecast_time": "datetime"})

    df = demand.merge(weather, on="datetime", how="left", validate="one_to_one")
    return df.merge(weather_forecast, on="datetime", how="left", validate="one_to_one")


def add_time_features(df):
    """Calendar features from UK local time."""
    df = df.copy()
    df["year"] = df["datetime_local"].dt.year
    df["month"] = df["datetime_local"].dt.month
    df["day_of_week"] = df["datetime_local"].dt.day_of_week
    df["hour"] = df["datetime_local"].dt.hour
    df["minute"] = df["datetime_local"].dt.minute
    df["day_of_year"] = df["datetime_local"].dt.day_of_year
    df["quarter"] = df["datetime_local"].dt.quarter
    df["is_weekend"] = (df["datetime_local"].dt.day_of_week >= 5).astype(int)

    # Holidays are generated up to next year so a live run never meets an uncovered date.
    uk_holidays = holidays.UnitedKingdom(
        subdiv="ENG", years=range(2009, pd.Timestamp.now().year + 2)
    )
    df["is_holiday"] = df["datetime_local"].dt.date.isin(set(uk_holidays.keys())).astype(int)
    return df


def build_dataset(
    end_date=None,
    benchmark_cutoff_time=BENCHMARK_CUTOFF_TIME,
    save=True,
    drop_missing_weather=True,
    with_benchmark=True,
):
    """Build the merged modelling dataset from the processed demand and weather files.

    The live forecast calls this with drop_missing_weather=False (the latest days have no ERA5
    weather yet but are still needed for the lags) and with_benchmark=False (not needed).
    """
    demand = add_settlement_timestamps(pd.read_parquet(DEMAND_PATH))
    weather = pd.read_parquet(WEATHER_PATH)
    weather_forecast = pd.read_parquet(WEATHER_FORECAST_PATH)

    if not EMBEDDED_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDED_PATH.name} not found. Run `python main.py` (or "
            "`python -m gb_demand_forecasting.data.embedded_forecast`) to download it."
        )

    embedded = pd.read_parquet(EMBEDDED_PATH)

    df = demand

    if with_benchmark:
        benchmark = build_day_ahead_benchmark(cutoff_time=benchmark_cutoff_time)
        df = add_benchmark(df, benchmark)

    df = add_embedded_forecast(df, embedded)
    df = add_weather(df, weather, weather_forecast)

    # Training cannot use rows without actual weather (the latest days, which ERA5 has not published).
    if drop_missing_weather:
        df = df.dropna(subset=["temperature_2m"])

    if end_date is not None:
        df = df.loc[df["local_date"] <= pd.Timestamp(end_date)]

    df = add_time_features(df)
    df = df.sort_values("datetime").set_index("datetime")

    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps in the merged dataset.")

    print(f"Merged shape: {df.shape}")
    print(f"Range: {df.index.min()} to {df.index.max()}")
    print(
        f"Prediction time {PREDICTION_TIME}; benchmark = newest NESO forecast published by {benchmark_cutoff_time} UK time on the day before"
    )
    if with_benchmark:
        print(f"Benchmark coverage: {df['elexon_forecast'].notna().mean():.1%} of periods")
    print(f"Embedded forecast coverage: {df['emb_solar_fc'].notna().mean():.1%} of periods")
    print(
        f"Missing demand: {int(df['ND'].isna().sum())} | imputed demand: {int(df['is_imputed'].sum())}"
    )

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=True)
        print(f"Saved: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    build_dataset()
