"""Forecast national demand for one day (default: tomorrow),
 using only what is known at 08:30 the day before.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import json

import mlflow
import mlflow.artifacts
import mlflow.sklearn
from mlflow import MlflowClient

from gb_demand_forecasting.data import build_dataset
from gb_demand_forecasting.data.embedded_forecast import fetch_embedded_forecast
from gb_demand_forecasting.data.weather.api import PINNED_PREVIOUS_RUNS_MODEL, fetch_previous_runs
from gb_demand_forecasting.data.weather.load_weather_previous_runs import (
    DEFAULT_LEAD_DAY,
    WEATHER_VARIABLES,
    average_weather_across_locations,
    expand_hourly_to_half_hourly,
    select_forecast_lead_day,
)
from gb_demand_forecasting.features.lags import build_features
from gb_demand_forecasting.features.model_inputs import select_model_features
from gb_demand_forecasting.features.weather import correct_forecast_weather
from gb_demand_forecasting.predictions.target_periods import get_target_periods

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECASTS_PATH = PROJECT_ROOT / "outputs" / "predictions" / "forecasts.parquet"

# Every feature looks back at least 2 days, so the target day's own neighbours are not needed.
LAST_USABLE_DAYS_BEFORE_TARGET = 2
MAX_EMBEDDED_AGE_HOURS = 8   # a guess: tune it after watching real issue times

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODEL_NAME = "gb_demand_blend"

CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"
CHAMPION_MODEL_PATH = CHAMPION_DIR / "model"
CHAMPION_INFO_PATH = CHAMPION_DIR / "model_info.json"



def load_model(model_name=MODEL_NAME, alias="champion", mlflow_db=MLFLOW_DB, champion_dir=CHAMPION_DIR):
    """Load the champion model: from models/champion if it exists, else from the MLflow registry.

    Returns (model, info, version) where version is a plain integer.
    """
    if champion_dir.exists():
        model = mlflow.sklearn.load_model(str(champion_dir / "model"))
        info = json.loads((champion_dir / "model_info.json").read_text())
        return model, info, int(info["model_version"])

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    version = MlflowClient().get_model_version_by_alias(name=model_name, alias=alias)
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
    info = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/model_info.json")
    return model, info, int(version.version)

def get_forecast_weather(target_date):
    """Get forecast weather for the target day in 30-minute intervals"""
    day = pd.Timestamp(target_date).date().isoformat()
    day_before = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).date().isoformat()   # UK summer days start at 23:00 UTC
    raw = fetch_previous_runs(day_before, day,
        variables=WEATHER_VARIABLES, 
        lead_days=(DEFAULT_LEAD_DAY,),
        models=PINNED_PREVIOUS_RUNS_MODEL
    )
    weather = expand_hourly_to_half_hourly(
        average_weather_across_locations(
            select_forecast_lead_day(raw, DEFAULT_LEAD_DAY)
            )
            )
    return weather.add_prefix("forecast_").rename(columns={"forecast_time": "datetime"})



def get_target_rows(target_date):
    """Generate rows for the target day."""
    day = pd.Timestamp(target_date).date().isoformat()
    target_data = get_target_periods(day).reset_index() 

    embedded = fetch_embedded_forecast(day, day)
    if len(embedded) != len(target_data):
        raise ValueError(f"Embedded forecast has {len(embedded)} periods for {day}, expected {len(target_data)}.")
    embedded = embedded.rename(columns={"settlement_date": "SETTLEMENT_DATE", "settlement_period": "SETTLEMENT_PERIOD"})
    target_data = build_dataset.add_embedded_forecast(target_data, embedded)

    target_data = target_data.merge(get_forecast_weather(target_date), on="datetime", how="left", validate="one_to_one")
    target_data = build_dataset.add_time_features(target_data)
    return target_data.set_index("datetime")



def prepare_prediction_inputs(target_date, historical_data=None):
    """Model inputs for the target day."""
    target_date = pd.Timestamp(target_date).normalize()

    if historical_data is None:
        historical_data = build_dataset.build_dataset(drop_missing_weather=False, with_benchmark=False, save=False)

    last_day = target_date - pd.Timedelta(days=LAST_USABLE_DAYS_BEFORE_TARGET)
    historical_data = historical_data[historical_data["local_date"] <= last_day]
    if historical_data["local_date"].max() < last_day:
        raise ValueError(f"History ends {historical_data['local_date'].max().date()}, need {last_day.date()}. Refresh the data first.")

    target_data = get_target_rows(target_date)
    for column in ("SETTLEMENT_DATE", "local_date"): 
        target_data[column] = target_data[column].astype(historical_data[column].dtype)

    combined_data = pd.concat([historical_data, target_data])
    combined_data = correct_forecast_weather(build_features(combined_data))
    day = combined_data[combined_data["local_date"] == target_date]
    return select_model_features(day, use_forecast_weather=True), target_data




def forecast_national_demand(target_date, historical_data=None, check_fresh=True):
    model, info, newest = load_model()
    X, rows = prepare_prediction_inputs(target_date, historical_data)

    if list(X.columns) != info["feature_columns"]:
        raise ValueError("The live features do not match the columns the model was trained on.")

    missing = X.isna().sum()
    if missing.any():
        raise ValueError(f"Missing feature values: {missing[missing > 0].to_dict()}")

    now = datetime.now(timezone.utc)
    if check_fresh:
        age = now - rows["emb_fc_issued_utc"].max().to_pydatetime()
        if age.total_seconds() > MAX_EMBEDDED_AGE_HOURS * 3600:
            raise ValueError(f"The newest embedded forecast is {age} old; NESO may not have published yet.")

    return pd.DataFrame({
        "target_date": pd.Timestamp(target_date).normalize(),
        "settlement_period": X["SETTLEMENT_PERIOD"].astype(int).to_numpy(),
        "datetime_utc": X.index,
        "forecast_mw": model.predict(X),
        "issued_at_utc": now,
        "embedded_forecast_issued_utc": rows["emb_fc_issued_utc"].max(),
        "model_trained_to": info["training_last_date"],
        "model_version": newest
    })




def save_forecast(new_forecast, forecast_file_path=FORECASTS_PATH, overwrite=False):
    """Append the day to the forecasts file."""
    path = Path(forecast_file_path)
    if path.exists():
        saved_forecasts = pd.read_parquet(path)
        if new_forecast["target_date"].iloc[0] in set(saved_forecasts["target_date"]):
            if not overwrite:
                raise ValueError(f"{new_forecast['target_date'].iloc[0].date()} is already forecast. Use --overwrite to replace it.")
            saved_forecasts = saved_forecasts[saved_forecasts["target_date"] != new_forecast["target_date"].iloc[0]]
        new_forecast = pd.concat([saved_forecasts, new_forecast], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_forecast.to_parquet(path, index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Forecast national demand for one day.")
    parser.add_argument("--date", default=None, help="Target day (YYYY-MM-DD). Default: tomorrow, UK time.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing forecast for that day.")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    today_uk = pd.Timestamp.now(tz="Europe/London").tz_localize(None).normalize()
    target = args.date or (today_uk + pd.Timedelta(days=1)).date().isoformat()
    forecast = forecast_national_demand(target, check_fresh = args.date is None)
    print(forecast[["settlement_period", "forecast_mw"]].describe().round(0))
    if not args.no_save:
        save_forecast(forecast, overwrite=args.overwrite)