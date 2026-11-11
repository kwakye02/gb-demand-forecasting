
import numpy as np
import pandas as pd

ACTUAL_WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                       "wind_speed_10m", "shortwave_radiation"]
FORECAST_WEATHER_COLS = ["forecast_temperature_2m", "forecast_relative_humidity_2m", "forecast_precipitation",
                         "forecast_wind_speed_10m", "forecast_shortwave_radiation"]

def correct_forecast_weather(data, lookback_days=60, delay_days=7):
    """Remove the recent bias of the forecast weather, using only days at least `delay_days` old."""
    data = data.copy()
    weather_bias_data = data.dropna(subset=ACTUAL_WEATHER_COLS + FORECAST_WEATHER_COLS)
    calendar_dates = pd.date_range(data["local_date"].min(), data["local_date"].max(), freq="D")
    cutoff_dates = data["local_date"] - pd.Timedelta(days=delay_days)

    def rolling_history_sum(values, dates):
        daily_totals = values.groupby(dates).sum().reindex(calendar_dates, fill_value=0)
        return daily_totals.rolling(lookback_days, min_periods=20).sum().reindex(cutoff_dates).to_numpy()

    observation_count = rolling_history_sum(
        pd.Series(1.0, index=weather_bias_data.index), weather_bias_data["local_date"]
    )

    for variable in ["temperature_2m", "relative_humidity_2m"]:
        average_bias = rolling_history_sum(
            weather_bias_data[f"forecast_{variable}"] - weather_bias_data[variable],
            weather_bias_data["local_date"],
        ) / observation_count
        data[f"forecast_{variable}"] -= np.nan_to_num(average_bias, nan=0.0, posinf=0.0, neginf=0.0)

    for variable in ["wind_speed_10m", "precipitation"]:
        correction_factor = (
            rolling_history_sum(weather_bias_data[variable], weather_bias_data["local_date"])
            / rolling_history_sum(weather_bias_data[f"forecast_{variable}"], weather_bias_data["local_date"])
        )
        data[f"forecast_{variable}"] *= np.nan_to_num(np.clip(correction_factor, 0.5, 2), nan=1)

    variable = "shortwave_radiation"
    utc_hours = data.index.hour
    radiation_factors = np.ones(len(data))

    for hour in range(24):
        hourly_history = weather_bias_data[weather_bias_data.index.hour == hour]
        actual_total = rolling_history_sum(hourly_history[variable], hourly_history["local_date"])
        forecast_total = rolling_history_sum(
            hourly_history[f"forecast_{variable}"], hourly_history["local_date"]
        )
        radiation_factors = np.where(
            utc_hours == hour,
            np.where(forecast_total > 500, actual_total / np.where(forecast_total > 0, forecast_total, 1), 1),
            radiation_factors,
        )

    data[f"forecast_{variable}"] *= np.clip(np.nan_to_num(radiation_factors, nan=1), 0.5, 2)
    return data

