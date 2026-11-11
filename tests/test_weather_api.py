import pandas as pd
import pytest
import requests

from gb_demand_forecasting.data.weather import api


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error",
                response=self,
            )
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


def weather_payload():
    return {
        "latitude": 51.5,
        "longitude": -0.1,
        "elevation": 25,
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "temperature_2m": [5.1, 5.0],
            "wind_speed_10m": [3.2, 3.3],
        },
    }


def test_build_location_params_uses_comma_separated_coordinates():
    locations = (
        api.WeatherLocation("one", 51.5, -0.1),
        api.WeatherLocation("two", 55.9, -3.2),
    )

    params = api.build_location_params(locations)

    assert params == {
        "latitude": "51.5,55.9",
        "longitude": "-0.1,-3.2",
    }


def test_validate_locations_rejects_duplicate_names():
    locations = (
        api.WeatherLocation("london", 51.5, -0.1),
        api.WeatherLocation("london", 52.0, -0.2),
    )

    with pytest.raises(ValueError, match="Duplicate location name"):
        api.validate_locations(locations)


def test_create_yearly_chunks_splits_date_range_by_calendar_year():
    chunks = api.create_yearly_chunks("2024-11-01", "2025-02-01")

    assert chunks == [
        ("2024-11-01", "2024-12-31"),
        ("2025-01-01", "2025-02-01"),
    ]


def test_create_monthly_chunks_splits_date_range_by_calendar_month():
    chunks = api.create_monthly_chunks("2022-01-15", "2022-03-02")

    assert chunks == [
        ("2022-01-15", "2022-01-31"),
        ("2022-02-01", "2022-02-28"),
        ("2022-03-01", "2022-03-02"),
    ]


def yearly_weather_frame(year):
    return pd.DataFrame(
        {
            "source": ["historical_weather", "historical_weather"],
            "location_id": [0, 0],
            "location_name": ["london", "london"],
            "requested_latitude": [51.5, 51.5],
            "requested_longitude": [-0.1, -0.1],
            "response_latitude": [51.5, 51.5],
            "response_longitude": [-0.1, -0.1],
            "elevation_m": [25, 25],
            "time": pd.to_datetime(
                [f"{year}-01-01T00:00", f"{year}-01-01T01:00"],
                utc=True,
            ),
            "temperature_2m": [5.1, 5.0],
        }
    )


def test_hourly_response_to_frame_adds_location_and_response_metadata():
    locations = (api.WeatherLocation("london", 51.5, -0.1),)

    dataframe = api.hourly_response_to_frame(
        weather_payload(),
        locations=locations,
        source="historical_weather",
    )

    assert list(dataframe["source"].unique()) == ["historical_weather"]
    assert list(dataframe["location_name"].unique()) == ["london"]
    assert dataframe.loc[0, "requested_latitude"] == 51.5
    assert dataframe.loc[0, "response_latitude"] == 51.5
    assert dataframe.loc[0, "elevation_m"] == 25
    assert dataframe.shape == (2, 11)
    assert pd.api.types.is_datetime64_any_dtype(dataframe["time"])
    assert str(dataframe["time"].dt.tz) == "UTC"


def test_hourly_response_to_frame_rejects_response_location_mismatch():
    locations = (
        api.WeatherLocation("london", 51.5, -0.1),
        api.WeatherLocation("edinburgh", 55.9, -3.2),
    )

    with pytest.raises(ValueError, match="does not match"):
        api.hourly_response_to_frame(
            weather_payload(),
            locations=locations,
            source="historical_weather",
        )


def test_fetch_historical_weather_uses_archive_endpoint_and_era5_model():
    session = FakeSession(weather_payload())
    locations = (api.WeatherLocation("london", 51.5, -0.1),)

    dataframe = api.fetch_historical_weather(
        start_date="2025-01-01",
        end_date="2025-01-01",
        locations=locations,
        hourly=("temperature_2m",),
        session=session,
    )

    call = session.calls[0]
    assert call["url"] == api.HISTORICAL_WEATHER_URL
    assert call["params"]["start_date"] == "2025-01-01"
    assert call["params"]["end_date"] == "2025-01-01"
    assert call["params"]["hourly"] == "temperature_2m"
    assert call["params"]["timezone"] == "GMT"
    assert call["params"]["models"] == "era5"
    assert len(dataframe) == 2


def test_fetch_live_forecast_uses_forecast_endpoint_without_current_weather():
    session = FakeSession(weather_payload())
    locations = (api.WeatherLocation("london", 51.5, -0.1),)

    api.fetch_live_forecast(
        forecast_days=3,
        locations=locations,
        hourly=("temperature_2m",),
        session=session,
    )

    call = session.calls[0]
    assert call["url"] == api.FORECAST_URL
    assert call["params"]["forecast_days"] == 3
    assert "current" not in call["params"]


def test_validate_weather_frame_rejects_duplicate_rows():
    dataframe = pd.DataFrame(
        {
            "source": ["historical_weather", "historical_weather"],
            "location_name": ["london", "london"],
            "time": pd.to_datetime(
                ["2025-01-01T00:00", "2025-01-01T00:00"],
                utc=True,
            ),
            "temperature_2m": [5.1, 5.1],
        }
    )

    with pytest.raises(ValueError, match="duplicate weather rows"):
        api.validate_weather_frame(
            dataframe,
            expected_hourly=("temperature_2m",),
        )


def test_validate_weather_frame_accepts_single_variable_string():
    dataframe = yearly_weather_frame(2025)

    result = api.validate_weather_frame(
        dataframe,
        expected_hourly="temperature_2m",
    )

    assert result is dataframe


def test_fetch_chunked_historical_weather_saves_and_reuses_chunks(monkeypatch, tmp_path):
    calls = []
    locations = (api.WeatherLocation("london", 51.5, -0.1),)

    def fake_fetch_historical_weather(
        start_date,
        end_date,
        locations,
        hourly,
        model,
        session,
    ):
        calls.append((start_date, end_date))
        return yearly_weather_frame(start_date[:4])

    monkeypatch.setattr(api, "fetch_historical_weather", fake_fetch_historical_weather)

    first_result = api.fetch_chunked_historical_weather(
        start_date="2024-01-01",
        end_date="2025-01-01",
        locations=locations,
        hourly=("temperature_2m",),
        chunk_cache_directory=tmp_path,
        sleep_seconds=0,
    )

    assert len(first_result) == 4
    assert calls == [
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-01-01"),
    ]
    assert (tmp_path / "gb_historical_weather_2024-01-01_2024-12-31.parquet").exists()
    assert (tmp_path / "gb_historical_weather_2025-01-01_2025-01-01.parquet").exists()

    calls.clear()

    second_result = api.fetch_chunked_historical_weather(
        start_date="2024-01-01",
        end_date="2025-01-01",
        locations=locations,
        hourly=("temperature_2m",),
        chunk_cache_directory=tmp_path,
        sleep_seconds=0,
    )

    assert len(second_result) == 4
    assert calls == []


def test_build_output_path_uses_parquet_suffix():
    output_path = api.build_output_path(
        "historical_weather",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert output_path.name == "gb_historical_weather_2025-01-01_2025-01-31.parquet"


def test_save_weather_frame_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match=".parquet or .csv"):
        api.save_weather_frame(pd.DataFrame({"x": [1]}), tmp_path / "weather.txt")


def test_request_open_meteo_raises_on_api_error():
    session = FakeSession({"error": True, "reason": "bad variable"})

    with pytest.raises(RuntimeError, match="bad variable"):
        api.request_open_meteo(api.FORECAST_URL, {}, session=session)


def test_request_open_meteo_retries_after_rate_limit(monkeypatch):
    sleeps = []
    session = SequenceSession(
        [
            FakeResponse({}, status_code=429, headers={"Retry-After": "0"}),
            FakeResponse(weather_payload()),
        ]
    )
    monkeypatch.setattr(api.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload = api.request_open_meteo(
        api.FORECAST_URL,
        {},
        session=session,
        rate_limit_wait_seconds=0,
    )

    assert payload == weather_payload()
    assert len(session.calls) == 2
    assert sleeps == [0]


def test_request_open_meteo_retries_after_connection_timeout(monkeypatch):
    sleeps = []

    class TimeoutThenSuccessSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params, timeout):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("read timed out")
            return FakeResponse(weather_payload())

    session = TimeoutThenSuccessSession()
    monkeypatch.setattr(api.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload = api.request_open_meteo(
        api.FORECAST_URL,
        {},
        session=session,
        rate_limit_wait_seconds=0,
    )

    assert payload == weather_payload()
    assert session.calls == 2
    assert sleeps == [0]
