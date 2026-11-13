import pandas as pd
import pytest

from gb_demand_forecasting.data.weather import load_historical_weather, load_weather_previous_runs


def two_city_frame(second_city_value):
    time = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame({
        "time": list(time) * 2,
        "temperature_2m": [10.0, 12.0, second_city_value, second_city_value],
    })


@pytest.mark.parametrize("module", [load_historical_weather, load_weather_previous_runs])
def test_average_across_locations_means_every_location(module):
    result = module.average_weather_across_locations(two_city_frame(20.0))

    assert result["temperature_2m"].tolist() == [15.0, 16.0]


@pytest.mark.parametrize("module", [load_historical_weather, load_weather_previous_runs])
def test_average_across_locations_uses_the_locations_that_have_data(module):
    frame = two_city_frame(20.0)
    frame.loc[2, "temperature_2m"] = None

    result = module.average_weather_across_locations(frame)

    assert result["temperature_2m"].tolist() == [10.0, 16.0]


@pytest.mark.parametrize("module", [load_historical_weather, load_weather_previous_runs])
def test_average_across_locations_keeps_timestamps_missing_everywhere(module):
    frame = two_city_frame(None)
    frame["temperature_2m"] = [None, 12.0, None, 14.0]

    result = module.average_weather_across_locations(frame)

    assert pd.isna(result["temperature_2m"].iloc[0])
    assert result["temperature_2m"].iloc[1] == 13.0


def test_download_previous_runs_uses_the_pinned_model_and_its_own_folder(monkeypatch, tmp_path):
    from gb_demand_forecasting.data.weather import api

    seen = {}

    def fake_chunks(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(api, "fetch_previous_runs_in_chunks", fake_chunks)

    with pytest.raises(RuntimeError):
        api.download_previous_runs("2026-01-01", "2026-01-31", session=object())

    assert seen["models"] == api.PINNED_PREVIOUS_RUNS_MODEL
    assert seen["chunk_cache_directory"].name == f"chunks_{api.PINNED_PREVIOUS_RUNS_MODEL}"
