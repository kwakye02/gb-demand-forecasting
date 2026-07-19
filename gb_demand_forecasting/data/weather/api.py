import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import time

import pandas as pd
import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_WEATHER_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "open_meteo"
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "open_meteo"

DEFAULT_TIMEZONE = "GMT"
DEFAULT_HISTORICAL_MODEL = "era5"
DEFAULT_CHUNK_DELAY_SECONDS = 60
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 90
DEFAULT_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
)


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float


GB_WEATHER_LOCATIONS = (
    WeatherLocation("london", 51.5072, -0.1276),
    WeatherLocation("birmingham", 52.4862, -1.8904),
    WeatherLocation("manchester", 53.4808, -2.2426),
    WeatherLocation("leeds", 53.8008, -1.5491),
    WeatherLocation("bristol", 51.4545, -2.5879),
    WeatherLocation("cardiff", 51.4816, -3.1791),
    WeatherLocation("newcastle", 54.9783, -1.6178),
    WeatherLocation("edinburgh", 55.9533, -3.1883),
    WeatherLocation("glasgow", 55.8642, -4.2518),
)


def _comma_join(values):
    return ",".join(str(value) for value in values)


def _format_variables(values):
    if values is None:
        return None

    if isinstance(values, str):
        return values

    return _comma_join(values)


def _normalise_variables(values):
    if values is None:
        return ()

    if isinstance(values, str):
        return tuple(value.strip() for value in values.split(",") if value.strip())

    return tuple(values)


def validate_locations(locations):
    if not locations:
        raise ValueError("At least one location is required.")

    names = set()

    for location in locations:
        if location.name in names:
            raise ValueError(f"Duplicate location name: {location.name}")

        if not -90 <= location.latitude <= 90:
            raise ValueError(f"Invalid latitude for {location.name}.")

        if not -180 <= location.longitude <= 180:
            raise ValueError(f"Invalid longitude for {location.name}.")

        names.add(location.name)


def build_location_params(locations=GB_WEATHER_LOCATIONS):
    validate_locations(locations)

    return {
        "latitude": _comma_join(location.latitude for location in locations),
        "longitude": _comma_join(location.longitude for location in locations),
    }


def build_weather_params(
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    timezone=DEFAULT_TIMEZONE,
    **extra_params,
):
    params = {
        **build_location_params(locations),
        "hourly": _format_variables(hourly),
        "timezone": timezone,
        "timeformat": "iso8601",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    params.update(
        {key: value for key, value in extra_params.items() if value is not None}
    )
    return params


def create_yearly_chunks(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(
            pd.Timestamp(
                year=current_start.year,
                month=12,
                day=31,
            ),
            end,
        )

        chunks.append(
            (
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
        )

        current_start = current_end + pd.Timedelta(days=1)

    return chunks


def create_monthly_chunks(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    chunks = []
    current_start = start

    while current_start <= end:
        current_end = min(
            current_start + pd.offsets.MonthEnd(0),
            end,
        )

        chunks.append(
            (
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
        )

        current_start = current_end + pd.Timedelta(days=1)

    return chunks


def create_date_chunks(start_date, end_date, frequency="year"):
    if frequency == "year":
        return create_yearly_chunks(start_date, end_date)

    if frequency == "month":
        return create_monthly_chunks(start_date, end_date)

    raise ValueError("frequency must be 'year' or 'month'.")


def create_request_session(
    cache_path=DEFAULT_CACHE_PATH,
    expire_after=None,
    retries=5,
    backoff_factor=0.2,
):
    try:
        import requests_cache
        from retry_requests import retry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install requests-cache and retry-requests to use cached Open-Meteo sessions."
        ) from exc

    cached_session = requests_cache.CachedSession(
        str(cache_path),
        expire_after=expire_after,
    )

    return retry(
        cached_session,
        retries=retries,
        backoff_factor=backoff_factor,
    )


def request_open_meteo(
    url,
    params,
    session=None,
    timeout=180,
    max_attempts=5,
    rate_limit_wait_seconds=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
):
    client = session or requests

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < max_attempts:
                wait_seconds = rate_limit_wait_seconds * attempt
                print(
                    "Open-Meteo connection/read timeout. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_attempts}."
                )
                time.sleep(wait_seconds)
                continue
            raise exc

        status_code = getattr(response, "status_code", None)

        if status_code == 429 and attempt < max_attempts:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            wait_seconds = (
                int(retry_after)
                if retry_after and retry_after.isdigit()
                else rate_limit_wait_seconds * attempt
            )
            print(
                "Open-Meteo rate limit reached. "
                f"Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_attempts}."
            )
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError:
            if status_code and status_code >= 500 and attempt < max_attempts:
                wait_seconds = rate_limit_wait_seconds * attempt
                print(
                    "Open-Meteo server error. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_attempts}."
                )
                time.sleep(wait_seconds)
                continue
            raise

        payload = response.json()

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(payload.get("reason", "Open-Meteo request failed."))

        return payload

    response.raise_for_status()
    return response.json()


def hourly_response_to_frame(payload, locations=GB_WEATHER_LOCATIONS, source=None):
    validate_locations(locations)
    responses = payload if isinstance(payload, list) else [payload]

    if len(responses) != len(locations):
        raise ValueError(
            "The number of Open-Meteo responses does not match "
            f"the number of requested locations: "
            f"{len(responses)} responses for {len(locations)} locations."
        )

    frames = []

    for location_id, (response, location) in enumerate(zip(responses, locations)):
        hourly = response.get("hourly")

        if not hourly:
            raise ValueError(f"No hourly data returned for {location.name}.")

        dataframe = pd.DataFrame(hourly)

        if "time" not in dataframe.columns:
            raise ValueError(f"No time column returned for {location.name}.")

        dataframe["time"] = pd.to_datetime(
            dataframe["time"],
            utc=True,
        )

        dataframe.insert(0, "source", source)
        dataframe.insert(1, "location_id", location_id)
        dataframe.insert(2, "location_name", location.name)
        dataframe.insert(3, "requested_latitude", location.latitude)
        dataframe.insert(4, "requested_longitude", location.longitude)
        dataframe.insert(5, "response_latitude", response.get("latitude"))
        dataframe.insert(6, "response_longitude", response.get("longitude"))
        dataframe.insert(7, "elevation_m", response.get("elevation"))
        frames.append(dataframe)

    return pd.concat(frames, ignore_index=True)


def validate_weather_frame(dataframe, expected_hourly=DEFAULT_HOURLY_VARIABLES):
    if dataframe.empty:
        raise ValueError("Weather DataFrame is empty.")

    required_columns = {
        "source",
        "location_name",
        "time",
        *_normalise_variables(expected_hourly),
    }

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    duplicates = dataframe.duplicated(
        subset=["source", "location_name", "time"]
    )

    if duplicates.any():
        raise ValueError(f"Found {duplicates.sum():,} duplicate weather rows.")

    for location_name, group in dataframe.groupby("location_name"):
        intervals = (
            group.sort_values("time")["time"]
            .diff()
            .dropna()
        )

        invalid = intervals != pd.Timedelta(hours=1)

        if invalid.any():
            print(
                f"Warning: {invalid.sum():,} non-hourly intervals for {location_name}."
            )

    return dataframe


def fetch_historical_weather(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    model=DEFAULT_HISTORICAL_MODEL,
    session=None,
):
    params = build_weather_params(
        locations=locations,
        hourly=hourly,
        start_date=start_date,
        end_date=end_date,
        models=model,
    )
    payload = request_open_meteo(HISTORICAL_WEATHER_URL, params, session=session)
    return hourly_response_to_frame(
        payload,
        locations=locations,
        source="historical_weather",
    )


def fetch_chunked_historical_weather(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    model=DEFAULT_HISTORICAL_MODEL,
    session=None,
    chunk_cache_directory=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
):
    frames = []
    chunks = create_date_chunks(start_date, end_date, frequency="year")

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_path = None

        if chunk_cache_directory:
            chunk_path = Path(chunk_cache_directory) / build_chunk_output_path(
                "historical_weather",
                chunk_start,
                chunk_end,
            ).name

        if chunk_path and chunk_path.exists():
            print(f"Using cached chunk {chunk_path.name}")
            chunk = read_weather_frame(chunk_path)
        else:
            print(f"Downloading {chunk_start} to {chunk_end}")

            chunk = fetch_historical_weather(
                start_date=chunk_start,
                end_date=chunk_end,
                locations=locations,
                hourly=hourly,
                model=model,
                session=session,
            )

            if chunk.empty:
                raise RuntimeError(f"No data returned for {chunk_start} to {chunk_end}.")

            validate_weather_frame(chunk, expected_hourly=hourly)

            if chunk_path:
                save_weather_frame(chunk, chunk_path)
                print(f"Saved chunk to {chunk_path}")

            if sleep_seconds > 0 and chunk_index < len(chunks) - 1:
                print(f"Waiting {sleep_seconds} seconds before next chunk.")
                time.sleep(sleep_seconds)

        frames.append(chunk)

    dataframe = pd.concat(frames, ignore_index=True)

    return (
        dataframe
        .sort_values(["location_name", "time"])
        .drop_duplicates(["location_name", "time"])
        .reset_index(drop=True)
    )


def fetch_historical_forecast(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    session=None,
):
    params = build_weather_params(
        locations=locations,
        hourly=hourly,
        start_date=start_date,
        end_date=end_date,
    )
    payload = request_open_meteo(HISTORICAL_FORECAST_URL, params, session=session)
    return hourly_response_to_frame(
        payload,
        locations=locations,
        source="historical_forecast",
    )


def fetch_chunked_historical_forecast(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    session=None,
    chunk_cache_directory=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
    chunk_frequency="month",
):
    frames = []
    chunks = create_date_chunks(start_date, end_date, frequency=chunk_frequency)

    for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_path = None

        if chunk_cache_directory:
            chunk_path = Path(chunk_cache_directory) / build_chunk_output_path(
                "historical_forecast",
                chunk_start,
                chunk_end,
            ).name

        if chunk_path and chunk_path.exists():
            print(f"Using cached chunk {chunk_path.name}")
            chunk = read_weather_frame(chunk_path)
        else:
            print(f"Downloading {chunk_start} to {chunk_end}")

            chunk = fetch_historical_forecast(
                start_date=chunk_start,
                end_date=chunk_end,
                locations=locations,
                hourly=hourly,
                session=session,
            )

            if chunk.empty:
                raise RuntimeError(f"No data returned for {chunk_start} to {chunk_end}.")

            validate_weather_frame(chunk, expected_hourly=hourly)

            if chunk_path:
                save_weather_frame(chunk, chunk_path)
                print(f"Saved chunk to {chunk_path}")

            if sleep_seconds > 0 and chunk_index < len(chunks) - 1:
                print(f"Waiting {sleep_seconds} seconds before next chunk.")
                time.sleep(sleep_seconds)

        frames.append(chunk)

    dataframe = pd.concat(frames, ignore_index=True)

    return (
        dataframe
        .sort_values(["location_name", "time"])
        .drop_duplicates(["location_name", "time"])
        .reset_index(drop=True)
    )


def fetch_live_forecast(
    forecast_days=7,
    locations=GB_WEATHER_LOCATIONS,
    hourly=DEFAULT_HOURLY_VARIABLES,
    session=None,
):
    params = build_weather_params(
        locations=locations,
        hourly=hourly,
        forecast_days=forecast_days,
    )
    payload = request_open_meteo(FORECAST_URL, params, session=session)
    return hourly_response_to_frame(
        payload,
        locations=locations,
        source="live_forecast",
    )


def save_weather_frame(dataframe, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        dataframe.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        dataframe.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end with .parquet or .csv.")

    return output_path


def read_weather_frame(input_path):
    input_path = Path(input_path)

    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)

    if input_path.suffix == ".csv":
        return pd.read_csv(input_path, parse_dates=["time"])

    raise ValueError("Input path must end with .parquet or .csv.")


def build_output_path(source, start_date=None, end_date=None):
    if source == "live_forecast":
        filename = f"gb_live_forecast_{date.today().isoformat()}.parquet"
    else:
        filename = f"gb_{source}_{start_date}_{end_date}.parquet"

    return RAW_WEATHER_DIRECTORY / source / filename


def build_chunk_output_path(source, start_date, end_date):
    filename = f"gb_{source}_{start_date}_{end_date}.parquet"
    return RAW_WEATHER_DIRECTORY / source / "chunks" / filename


def download_historical_weather(
    start_date,
    end_date,
    session=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
):
    session = session or create_request_session(expire_after=None)
    dataframe = fetch_chunked_historical_weather(
        start_date=start_date,
        end_date=end_date,
        session=session,
        chunk_cache_directory=RAW_WEATHER_DIRECTORY / "historical_weather" / "chunks",
        sleep_seconds=sleep_seconds,
    )
    validate_weather_frame(dataframe)

    output_path = build_output_path(
        "historical_weather",
        start_date=start_date,
        end_date=end_date,
    )
    save_weather_frame(dataframe, output_path)
    print(f"Saved {len(dataframe):,} rows to {output_path}")
    return output_path


def download_historical_forecast(
    start_date,
    end_date,
    session=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
    chunk_frequency="month",
):
    session = session or create_request_session(expire_after=None)
    dataframe = fetch_chunked_historical_forecast(
        start_date=start_date,
        end_date=end_date,
        session=session,
        chunk_cache_directory=RAW_WEATHER_DIRECTORY / "historical_forecast" / "chunks",
        sleep_seconds=sleep_seconds,
        chunk_frequency=chunk_frequency,
    )
    validate_weather_frame(dataframe)

    output_path = build_output_path(
        "historical_forecast",
        start_date=start_date,
        end_date=end_date,
    )
    save_weather_frame(dataframe, output_path)
    print(f"Saved {len(dataframe):,} rows to {output_path}")
    return output_path


def download_live_forecast(forecast_days=7, session=None):
    session = session or create_request_session(expire_after=3600)
    dataframe = fetch_live_forecast(forecast_days=forecast_days, session=session)
    validate_weather_frame(dataframe)

    output_path = build_output_path("live_forecast")
    save_weather_frame(dataframe, output_path)
    print(f"Saved {len(dataframe):,} rows to {output_path}")
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo weather data for Great Britain."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    historical_weather = subparsers.add_parser(
        "historical-weather",
        help="Download historical observed weather.",
    )
    historical_weather.add_argument("--start-date", required=True)
    historical_weather.add_argument("--end-date", required=True)
    historical_weather.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_CHUNK_DELAY_SECONDS,
        help="Seconds to wait between chunks.",
    )

    historical_forecast = subparsers.add_parser(
        "historical-forecast",
        help="Download historical forecast weather.",
    )
    historical_forecast.add_argument("--start-date", required=True)
    historical_forecast.add_argument("--end-date", required=True)
    historical_forecast.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_CHUNK_DELAY_SECONDS,
        help="Seconds to wait between chunks.",
    )
    historical_forecast.add_argument(
        "--chunk-frequency",
        choices=("month", "year"),
        default="month",
        help="Chunk size for historical forecast downloads. Monthly is safer for large requests.",
    )

    live_forecast = subparsers.add_parser(
        "live-forecast",
        help="Download current live forecast weather.",
    )
    live_forecast.add_argument("--forecast-days", type=int, default=7)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.command == "historical-weather":
        return download_historical_weather(
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_seconds=args.sleep_seconds,
        )

    if args.command == "historical-forecast":
        return download_historical_forecast(
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_seconds=args.sleep_seconds,
            chunk_frequency=args.chunk_frequency,
        )

    if args.command == "live-forecast":
        return download_live_forecast(forecast_days=args.forecast_days)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
