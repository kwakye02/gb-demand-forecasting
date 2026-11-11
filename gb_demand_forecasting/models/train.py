"""Fit the final CatBoost + Ridge blend on every available day.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from gb_demand_forecasting.features.lags import FEATURE_COLS, build_features
from gb_demand_forecasting.features.model_inputs import CAT_FEATURES, MODEL_COLS, select_model_features
from gb_demand_forecasting.features.weather import correct_forecast_weather
from gb_demand_forecasting.models.catboost_ridge_blend import CatBoostRidgeBlend, TRUSTED_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_PATH = PROJECT_ROOT / "data" / "processed" / "gb_demand_merged.parquet"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT = "GB_electricity_demand_forecasting"
MODEL_NAME = "gb_demand_blend"

TRAIN_START = "2020-01-01"


def training_features(end_date=None):
    """Merged data with every feature built, the same way the live forecast will build them."""
    df = pd.read_parquet(MERGED_PATH)
    df = correct_forecast_weather(build_features(df))
    df = df[df["SETTLEMENT_DATE"] >= TRAIN_START].dropna(subset=FEATURE_COLS + ["ND"])

    if end_date is not None:
        df = df[df["local_date"] <= pd.Timestamp(end_date)]

    return df


def recency_weights(index):
    """A day's weight halves for every year of age."""
    age_years = (index.max() - index).days / 365.25
    return np.asarray(0.5 ** age_years)


def train_model(end_date=None, tracking_uri=f"sqlite:///{MLFLOW_DB.as_posix()}"):
    df = training_features(end_date)
    X_train = select_model_features(df, use_forecast_weather=False)
    y_train = df["ND"]

    model = CatBoostRidgeBlend(cat_features=CAT_FEATURES)
    model.fit(X_train, y_train, sample_weight=recency_weights(df.index))

    fitted = model.predict(X_train)
    if not np.isfinite(fitted).all():
        raise ValueError("The fitted model produced non-finite predictions on its own training data.")

    info = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_first_date": str(df["local_date"].min().date()),
        "training_last_date": str(df["local_date"].max().date()),
        "training_rows": int(len(df)),
        "in_sample_mae_mw": round(float(np.abs(y_train - fitted).mean()), 1),
        "feature_columns": list(X_train.columns),
        "model_input_columns": MODEL_COLS,
        "ridge_weight": model.ridge_weight,
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="Final_CatBoost_Ridge_Blend") as run:
        mlflow.log_params({k: info[k] for k in ("training_first_date", "training_last_date",
                                                "training_rows", "ridge_weight")})
        mlflow.log_metric("in_sample_mae_mw", info["in_sample_mae_mw"])
        mlflow.log_dict(info, "model_info.json")
        mlflow.sklearn.log_model(model,name="model",
                                 serialization_format="skops",
                                 skops_trusted_types=TRUSTED_TYPES,
                                 registered_model_name=MODEL_NAME)

    print(json.dumps({k: v for k, v in info.items() if k != "feature_columns"}, indent=2))
    print(f"Logged run {run.info.run_id} and registered it as '{MODEL_NAME}'")
    return model, info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the final demand model on all available days.")
    parser.add_argument("--end-date", default=None, help="Last day to train on (YYYY-MM-DD). Default: all.")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{MLFLOW_DB.as_posix()}")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_model(args.end_date, args.tracking_uri)
