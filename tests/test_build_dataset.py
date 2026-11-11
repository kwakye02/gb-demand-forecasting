import pandas as pd
import pytest

from gb_demand_forecasting.data import build_dataset


def demand_frame(date, periods):
    return pd.DataFrame({
        "SETTLEMENT_DATE": [date] * len(periods),
        "SETTLEMENT_PERIOD": periods,
        "ND": [30000.0] * len(periods),
        "is_imputed": False,
    })


def test_timestamps_use_uk_local_time_and_convert_to_utc_in_summer():
    df = build_dataset.add_settlement_timestamps(demand_frame("2026-06-10", [1, 2]))

    assert df["local_time"].tolist() == ["00:00", "00:30"]
    # Period 1 starts at 00:00 BST, which is 23:00 UTC the previous day.
    assert df["datetime"].iloc[0] == pd.Timestamp("2026-06-09T23:00:00Z")


def test_timestamps_stay_unique_on_the_long_clock_change_day():
    df = build_dataset.add_settlement_timestamps(demand_frame("2026-10-25", list(range(1, 51))))

    assert df["datetime"].is_unique
    assert df["datetime"].iloc[-1] - df["datetime"].iloc[0] == pd.Timedelta(minutes=30 * 49)


def test_add_benchmark_keeps_one_row_per_settlement_period_and_leaves_gaps_empty():
    demand = demand_frame("2026-06-10", [1, 2])
    benchmark = pd.DataFrame({
        "settlement_date": [pd.Timestamp("2026-06-10")],
        "settlement_period": [1],
        "elexon_forecast_published_utc": [pd.Timestamp("2026-06-09T07:45:00Z")],
        "elexon_forecast": [29000.0],
    })
    demand["SETTLEMENT_DATE"] = pd.to_datetime(demand["SETTLEMENT_DATE"])

    merged = build_dataset.add_benchmark(demand, benchmark)

    assert len(merged) == 2
    assert merged["elexon_forecast"].iloc[0] == 29000.0
    assert pd.isna(merged["elexon_forecast"].iloc[1])


def test_add_weather_prefixes_forecast_columns_and_rejects_duplicate_timestamps():
    demand = pd.DataFrame({"datetime": pd.to_datetime(["2026-06-10T00:00Z", "2026-06-10T00:30Z"])})
    weather = pd.DataFrame({
        "time": pd.to_datetime(["2026-06-10T00:00Z", "2026-06-10T00:30Z"]),
        "temperature_2m": [10.0, 10.5],
    })
    forecast = weather.assign(temperature_2m=[11.0, 11.5])

    merged = build_dataset.add_weather(demand, weather, forecast)

    assert merged["temperature_2m"].tolist() == [10.0, 10.5]
    assert merged["forecast_temperature_2m"].tolist() == [11.0, 11.5]

    with pytest.raises(Exception):
        build_dataset.add_weather(demand, pd.concat([weather, weather]), forecast)


def test_time_features_flag_weekends_and_holidays_from_uk_local_date():
    df = pd.DataFrame({
        "datetime_local": pd.to_datetime(["2026-12-25 12:00", "2026-12-26 12:00", "2026-12-28 12:00"])
        .tz_localize("Europe/London")
    })

    features = build_dataset.add_time_features(df)

    assert features["is_holiday"].tolist() == [1, 1, 1]  # Christmas, Boxing Day, substitute holiday
    assert features["is_weekend"].tolist() == [0, 1, 0]  # Friday, Saturday, Monday
