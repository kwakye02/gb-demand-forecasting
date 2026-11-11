import numpy as np
import pandas as pd 
from gb_demand_forecasting.features.weather import ACTUAL_WEATHER_COLS, FORECAST_WEATHER_COLS

MODEL_COLS = ["SETTLEMENT_PERIOD", "day_of_week", "month", "is_holiday",
              "temperature_2m", "relative_humidity_2m", "precipitation",
              "shortwave_radiation", "wind_speed_10m", "lag_2d", "lag_7d",
              "xmas_period", "day_before_holiday", "day_after_holiday",
              "prev_6weeks_lag_noholiday", "lag7d_minus_prev_6weeks",
              "d_solar_7", "d_wind_7", "d_solar_base", "d_wind_base",
              "emb_solar_fc", "emb_wind_fc"]
CAT_FEATURES = ["SETTLEMENT_PERIOD", "day_of_week", "month", "dow_sp", "hol_sp"]


def add_temperature_time_features(X):
    """Heating and cooling degree days, their wind interaction, temperature squared, day-of-week x period."""
    X = X.copy()
    X["hdd"] = (15.5 - X["temperature_2m"]).clip(lower=0)
    X["hdd_x_wind"] = X["hdd"] * X["wind_speed_10m"]
    X["cdd"] = (X["temperature_2m"] - 20).clip(lower=0)
    X["temp2"] = X["temperature_2m"] ** 2
    X["dow_sp"] = X["day_of_week"].astype(str) + "_" + X["SETTLEMENT_PERIOD"].astype(str)
    return X


def add_holiday_period(X, holiday_col, period_col):
    """hol_sp: the settlement period on bank holidays, 0 on other days."""
    X = X.copy()
    X["hol_sp"] = np.where(X[holiday_col] == 1, X[period_col], 0)
    return X

def prepare_model_features(data, feature_cols, use_forecast_weather,
                           actual_weather_cols=ACTUAL_WEATHER_COLS,
                           forecast_weather_cols=FORECAST_WEATHER_COLS):
    """Select the model columns, using forecast weather (renamed to the actual names) or actual weather."""
    actual_to_forecast_names = dict(zip(actual_weather_cols, forecast_weather_cols))
    forecast_to_actual_names = {v: k for k, v in actual_to_forecast_names.items()}

    selected_cols = [
        actual_to_forecast_names.get(c, c) if use_forecast_weather else c
        for c in feature_cols
    ]
    return data[selected_cols].rename(columns=forecast_to_actual_names)

def select_model_features(data, use_forecast_weather, feature_cols=MODEL_COLS):
    """The full matrix the models are fitted on and predict from."""
    X = prepare_model_features(data, feature_cols, use_forecast_weather)
    X = add_holiday_period(add_temperature_time_features(X), "is_holiday", "SETTLEMENT_PERIOD")
    for col in CAT_FEATURES:
        X[col] = X[col].astype("category")
    return X