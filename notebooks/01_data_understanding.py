#!/usr/bin/env python
# coding: utf-8

"""
Exploratory script: runs the NESO demand data pipeline and inspects the result.
Core logic lives in gb_demand_forecasting/data/load_neso.py — this notebook is
for exploration only.
"""


from gb_demand_forecasting.data.demand.load_neso import run_neso_pipeline
from gb_demand_forecasting.data.weather.load_historical_weather import run_historical_weather_pipeline
from gb_demand_forecasting.data.weather.load_weather_forecast import run_weather_forecast_pipeline

print("**"*10 + "Running demand pipeline" + "**"*10)
print("\n")

combined_df = run_neso_pipeline(save=True)
print(combined_df.head())
print(combined_df.shape)
print(combined_df.isnull().sum())

weather_gb = run_historical_weather_pipeline(save=True)
print(weather_gb.head())
print(f"Shape of historical weather data: {weather_gb.shape}")
print("\n")


print("**"*10 + "Running weather forecast pipeline" + "**"*10)
print("\n")
weather_forecast_gb = run_weather_forecast_pipeline(save=True)
print(f"Shape of weather forecast data: {weather_forecast_gb.shape}")
print(weather_forecast_gb.head())
print("\n")

