"""
Download historical, archived forecast, and live weather data from Open-Meteo.

Weather data is collected for nine GB locations and saved in
Parquet files for processing and electricity demand forecasting.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

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


def _join_with_commas(values):
    return ",".join(str(value) for value in values)


def _format_variables(values):
    if values is None:
        return None

    if isinstance(values, str):
        return values

    return _join_with_commas(values)


def _normalise_variables(values):
    if values is None:
        return ()

    if isinstance(values, str):
        return tuple(value.strip() for value in values.split(",") if value.strip())

    return tuple(values)


def validate_locations(locations):
    if not locations:
        raise ValueError("At least one location is required.")

    location_names = set()

    for location in locations:
        if location.name in location_names:
            raise ValueError(f"Duplicate location name: {location.name}")

        if not -90 <= location.latitude <= 90:
            raise ValueError(f"Invalid latitude for {location.name}.")

        if not -180 <= location.longitude <= 180:
            raise ValueError(f"Invalid longitude for {location.name}.")

        location_names.add(location.name)


def build_location_params(locations=GB_WEATHER_LOCATIONS):
    validate_locations(locations)

    return {
        "latitude": _join_with_commas(location.latitude for location in locations),
        "longitude": _join_with_commas(location.longitude for location in locations),
    }


def build_weather_params(
    locations=GB_WEATHER_LOCATIONS,
    hourly_variables=DEFAULT_HOURLY_VARIABLES,
    timezone=DEFAULT_TIMEZONE,
    **extra_params,
):
    """Build the parameters for an Open-Meteo API request."""

    request_params = {
        **build_location_params(locations),
        "hourly": _format_variables(hourly_variables),
        "timezone": timezone,
        "timeformat": "iso8601",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    request_params.update({key: value for key, value in extra_params.items() if value is not None})
    return request_params


def get_yearly_date_ranges(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    date_ranges = []
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

        date_ranges.append(
            (
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
        )

        current_start = current_end + pd.Timedelta(days=1)

    return date_ranges


def get_monthly_date_ranges(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    date_ranges = []
    current_start = start

    while current_start <= end:
        current_end = min(
            current_start + pd.offsets.MonthEnd(0),
            end,
        )

        date_ranges.append(
            (
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
        )

        current_start = current_end + pd.Timedelta(days=1)

    return date_ranges


def get_date_ranges(start_date, end_date, frequency="year"):
    if frequency == "year":
        return get_yearly_date_ranges(start_date, end_date)

    if frequency == "month":
        return get_monthly_date_ranges(start_date, end_date)

    raise ValueError("frequency must be 'year' or 'month'.")


def create_cached_session(cache_path=DEFAULT_CACHE_PATH, expire_after=None):
    """Create a cached HTTP session; retries are handled in request_open_meteo."""
    try:
        import requests_cache
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install requests-cache to use cached Open-Meteo sessions."
        ) from exc

    return requests_cache.CachedSession(
        str(cache_path),
        expire_after=expire_after,
    )


def request_open_meteo(
    url,
    request_params,
    session=None,
    timeout=180,
    max_attempts=5,
    rate_limit_wait_seconds=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
):
    """Request Open-Meteo data with retries for temporary failures."""

    client = session or requests

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, params=request_params, timeout=timeout)
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

        try:
            response_data = response.json()
        except ValueError:
            cache = getattr(client, "cache", None)

            if cache is not None:
                cache.delete(urls=[response.url])

            if attempt < max_attempts:
                wait_seconds = rate_limit_wait_seconds * attempt
                print(
                    "Open-Meteo returned an incomplete response. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}/{max_attempts}."
                )
                time.sleep(wait_seconds)
                continue
            raise

        if isinstance(response_data, dict) and response_data.get("error"):
            raise RuntimeError(response_data.get("reason", "Open-Meteo request failed."))

        return response_data

    response.raise_for_status()
    return response.json()


def parse_hourly_weather_response(response_data, locations=GB_WEATHER_LOCATIONS, source=None):
    """Convert hourly API responses into a weather DataFrame."""

    validate_locations(locations)
    location_responses = response_data if isinstance(response_data, list) else [response_data]

    if len(location_responses) != len(locations):
        raise ValueError(
            "The number of Open-Meteo responses does not match "
            f"the number of requested locations: "
            f"{len(location_responses)} responses for {len(locations)} locations."
        )

    weather_frames = []

    for location_index, (response, location) in enumerate(zip(location_responses, locations)):
        hourly_data = response.get("hourly")

        if not hourly_data:
            raise ValueError(f"No hourly data returned for {location.name}.")

        weather_data = pd.DataFrame(hourly_data)

        if "time" not in weather_data.columns:
            raise ValueError(f"No time column returned for {location.name}.")

        weather_data["time"] = pd.to_datetime(
            weather_data["time"],
            utc=True,
        )

        weather_data.insert(0, "source", source)
        weather_data.insert(1, "location_id", location_index)
        weather_data.insert(2, "location_name", location.name)
        weather_data.insert(3, "requested_latitude", location.latitude)
        weather_data.insert(4, "requested_longitude", location.longitude)
        weather_data.insert(5, "response_latitude", response.get("latitude"))
        weather_data.insert(6, "response_longitude", response.get("longitude"))
        weather_data.insert(7, "elevation_m", response.get("elevation"))
        weather_frames.append(weather_data)

    return pd.concat(weather_frames, ignore_index=True)


def validate_weather_data(weather_data, expected_hourly=DEFAULT_HOURLY_VARIABLES):
    """Check weather data columns, duplicates and hourly timestamps."""

    if weather_data.empty:
        raise ValueError("Weather DataFrame is empty.")

    required_columns = {
        "source",
        "location_name",
        "time",
        *_normalise_variables(expected_hourly),
    }

    missing_columns = required_columns.difference(weather_data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    duplicate_rows = weather_data.duplicated(subset=["source", "location_name", "time"])

    if duplicate_rows.any():
        raise ValueError(f"Found {duplicate_rows.sum():,} duplicate weather rows.")

    for location_name, group in weather_data.groupby("location_name"):
        time_intervals = group.sort_values("time")["time"].diff().dropna()

        invalid_intervals = time_intervals != pd.Timedelta(hours=1)

        if invalid_intervals.any():
            raise ValueError(
                f"{invalid_intervals.sum():,} non-hourly intervals for {location_name}."
            )

    variable_columns = sorted(required_columns.difference({"source", "location_name", "time"}))
    missing_counts = weather_data[variable_columns].isna().sum()

    if (missing_counts == len(weather_data)).any():
        empty = missing_counts[missing_counts == len(weather_data)].index.tolist()
        raise ValueError(f"Columns with no values at all: {empty}")

    if missing_counts.any():
        rows_with_missing = weather_data[weather_data[variable_columns].isna().any(axis=1)]
        print(
            f"Warning: {int(missing_counts.sum()):,} missing values "
            f"({rows_with_missing['time'].min():%Y-%m-%d} to "
            f"{rows_with_missing['time'].max():%Y-%m-%d}) in: "
            f"{missing_counts[missing_counts > 0].to_dict()}"
        )

    return weather_data


def load_valid_cached_chunk(chunk_file_path, expected_hourly):
    """Load a saved chunk, or return None if it lacks the expected columns or is invalid."""
    try:
        return validate_weather_data(load_weather_data(chunk_file_path), expected_hourly)
    except ValueError as error:
        print(f"Ignoring cached chunk {chunk_file_path.name}: {error}")
        return None


def fetch_historical_weather(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly_variables=DEFAULT_HOURLY_VARIABLES,
    model=DEFAULT_HISTORICAL_MODEL,
    session=None,
):
    """Download historical weather observations from Open-Meteo."""

    request_params = build_weather_params(
        locations=locations,
        hourly_variables=hourly_variables,
        start_date=start_date,
        end_date=end_date,
        models=model,
    )
    response_data = request_open_meteo(HISTORICAL_WEATHER_URL, request_params, session=session)
    return parse_hourly_weather_response(
        response_data,
        locations=locations,
        source="historical_weather",
    )


def fetch_historical_weather_in_chunks(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    hourly_variables=DEFAULT_HOURLY_VARIABLES,
    model=DEFAULT_HISTORICAL_MODEL,
    session=None,
    chunk_cache_directory=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
):
    """Download historical weather in yearly date ranges."""

    weather_frames = []
    date_ranges = get_date_ranges(start_date, end_date, frequency="year")

    for chunk_index, (chunk_start, chunk_end) in enumerate(date_ranges):
        chunk_file_path = None

        if chunk_cache_directory:
            chunk_file_path = (
                Path(chunk_cache_directory)
                / get_chunk_file_path(
                    "historical_weather",
                    chunk_start,
                    chunk_end,
                ).name
            )

        weather_chunk = None

        if chunk_file_path and chunk_file_path.exists():
            weather_chunk = load_valid_cached_chunk(chunk_file_path, hourly_variables)

            if weather_chunk is not None:
                print(f"Using cached chunk {chunk_file_path.name}")

        if weather_chunk is None:
            print(f"Downloading {chunk_start} to {chunk_end}")

            weather_chunk = fetch_historical_weather(
                start_date=chunk_start,
                end_date=chunk_end,
                locations=locations,
                hourly_variables=hourly_variables,
                model=model,
                session=session,
            )

            if weather_chunk.empty:
                raise RuntimeError(f"No data returned for {chunk_start} to {chunk_end}.")

            validate_weather_data(weather_chunk, expected_hourly=hourly_variables)

            if chunk_file_path:
                save_weather_data(weather_chunk, chunk_file_path)
                print(f"Saved chunk to {chunk_file_path}")

            if sleep_seconds > 0 and chunk_index < len(date_ranges) - 1:
                print(f"Waiting {sleep_seconds} seconds before next chunk.")
                time.sleep(sleep_seconds)

        weather_frames.append(weather_chunk)

    weather_data = pd.concat(weather_frames, ignore_index=True)

    return (
        weather_data.sort_values(["location_name", "time"])
        .drop_duplicates(["location_name", "time"])
        .reset_index(drop=True)
    )


def fetch_live_forecast(
    forecast_days=7,
    locations=GB_WEATHER_LOCATIONS,
    hourly_variables=DEFAULT_HOURLY_VARIABLES,
    session=None,
    models=None,
):
    request_params = build_weather_params(
        locations=locations,
        hourly_variables=hourly_variables,
        forecast_days=forecast_days,
        models=models,
    )
    response_data = request_open_meteo(FORECAST_URL, request_params, session=session)
    return parse_hourly_weather_response(
        response_data,
        locations=locations,
        source="live_forecast",
    )


# rain and snowfall are largely empty in the Previous Runs archive, so they are left out.
PREVIOUS_RUNS_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
)
DEFAULT_PREVIOUS_RUNS_LEAD_DAYS = (1, 2)
# ECMWF IFS 0.25° was selected using the 2025 weather comparison
# and is fixed across the historical forecast evaluation.

# Known data gap: ECMWF cloud cover is missing from
# 17–22 April 2026 in the Open-Meteo Previous Runs archive.
# Exclude affected evaluation dates when cloud cover is required.

PINNED_PREVIOUS_RUNS_MODEL = "ecmwf_ifs025"


def get_previous_run_columns(
    variables=PREVIOUS_RUNS_VARIABLES,
    lead_days=DEFAULT_PREVIOUS_RUNS_LEAD_DAYS,
):
    return tuple(
        f"{variable}_previous_day{lead_day}" for variable in variables for lead_day in lead_days
    )


def fetch_previous_runs(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    variables=PREVIOUS_RUNS_VARIABLES,
    lead_days=DEFAULT_PREVIOUS_RUNS_LEAD_DAYS,
    session=None,
    models=None,
):
    """Download archived forecasts for the requested lead times."""

    request_params = build_weather_params(
        locations=locations,
        hourly_variables=get_previous_run_columns(variables, lead_days),
        start_date=start_date,
        end_date=end_date,
        models=models,
    )
    response_data = request_open_meteo(PREVIOUS_RUNS_URL, request_params, session=session)
    return parse_hourly_weather_response(
        response_data,
        locations=locations,
        source="previous_runs",
    )


def fetch_previous_runs_in_chunks(
    start_date,
    end_date,
    locations=GB_WEATHER_LOCATIONS,
    variables=PREVIOUS_RUNS_VARIABLES,
    lead_days=DEFAULT_PREVIOUS_RUNS_LEAD_DAYS,
    session=None,
    chunk_cache_directory=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
    models=None,
):
    """Download archived weather forecasts in monthly date ranges."""

    weather_frames = []
    columns = get_previous_run_columns(variables, lead_days)
    date_ranges = get_date_ranges(start_date, end_date, frequency="month")

    for chunk_index, (chunk_start, chunk_end) in enumerate(date_ranges):
        chunk_file_path = None

        if chunk_cache_directory:
            chunk_file_path = (
                Path(chunk_cache_directory)
                / get_chunk_file_path(
                    "previous_runs",
                    chunk_start,
                    chunk_end,
                ).name
            )

        weather_chunk = None

        if chunk_file_path and chunk_file_path.exists():
            weather_chunk = load_valid_cached_chunk(chunk_file_path, columns)

            if weather_chunk is not None:
                print(f"Using cached chunk {chunk_file_path.name}")

        if weather_chunk is None:
            print(f"Downloading {chunk_start} to {chunk_end}")

            weather_chunk = fetch_previous_runs(
                start_date=chunk_start,
                end_date=chunk_end,
                locations=locations,
                variables=variables,
                lead_days=lead_days,
                session=session,
                models=models,
            )

            if weather_chunk.empty:
                raise RuntimeError(f"No data returned for {chunk_start} to {chunk_end}.")

            validate_weather_data(weather_chunk, expected_hourly=columns)

            if chunk_file_path:
                save_weather_data(weather_chunk, chunk_file_path)
                print(f"Saved chunk to {chunk_file_path}")

            if sleep_seconds > 0 and chunk_index < len(date_ranges) - 1:
                print(f"Waiting {sleep_seconds} seconds before next chunk.")
                time.sleep(sleep_seconds)

        weather_frames.append(weather_chunk)

    weather_data = pd.concat(weather_frames, ignore_index=True)

    return (
        weather_data.sort_values(["location_name", "time"])
        .drop_duplicates(["location_name", "time"])
        .reset_index(drop=True)
    )


def save_weather_data(weather_data, output_path):
    """Save weather data as a Parquet or CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        weather_data.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        weather_data.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path must end with .parquet or .csv.")

    return output_path


def load_weather_data(input_path):
    """Load weather data from a Parquet or CSV file."""

    input_path = Path(input_path)

    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)

    if input_path.suffix == ".csv":
        return pd.read_csv(input_path, parse_dates=["time"])

    raise ValueError("Input path must end with .parquet or .csv.")


def get_output_file_path(source, start_date=None, end_date=None):
    if source == "live_forecast":
        filename = f"gb_live_forecast_{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H%M}Z.parquet"
    else:
        filename = f"gb_{source}_{start_date}_{end_date}.parquet"

    return RAW_WEATHER_DIRECTORY / source / filename


def get_previous_runs_chunk_directory(model=None):
    """Chunk folder for previous-runs data; one folder per pinned model."""
    name = f"chunks_{model}" if model else "chunks"
    return RAW_WEATHER_DIRECTORY / "previous_runs" / name


def get_chunk_file_path(source, start_date, end_date):
    filename = f"gb_{source}_{start_date}_{end_date}.parquet"
    return RAW_WEATHER_DIRECTORY / source / "chunks" / filename


def download_historical_weather(
    start_date,
    end_date,
    session=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
):
    session = session or create_cached_session(expire_after=None)
    weather_data = fetch_historical_weather_in_chunks(
        start_date=start_date,
        end_date=end_date,
        session=session,
        chunk_cache_directory=RAW_WEATHER_DIRECTORY / "historical_weather" / "chunks",
        sleep_seconds=sleep_seconds,
    )
    validate_weather_data(weather_data)

    output_path = get_output_file_path(
        "historical_weather",
        start_date=start_date,
        end_date=end_date,
    )
    save_weather_data(weather_data, output_path)
    print(f"Saved {len(weather_data):,} rows to {output_path}")
    return output_path


def download_previous_runs(
    start_date,
    end_date,
    lead_days=DEFAULT_PREVIOUS_RUNS_LEAD_DAYS,
    session=None,
    sleep_seconds=DEFAULT_CHUNK_DELAY_SECONDS,
    models=PINNED_PREVIOUS_RUNS_MODEL,
):
    session = session or create_cached_session(expire_after=None)
    weather_data = fetch_previous_runs_in_chunks(
        start_date=start_date,
        end_date=end_date,
        lead_days=lead_days,
        session=session,
        chunk_cache_directory=get_previous_runs_chunk_directory(models),
        sleep_seconds=sleep_seconds,
        models=models,
    )
    validate_weather_data(
        weather_data,
        expected_hourly=get_previous_run_columns(lead_days=lead_days),
    )

    output_path = get_output_file_path(
        "previous_runs",
        start_date=start_date,
        end_date=end_date,
    )
    save_weather_data(weather_data, output_path)
    print(f"Saved {len(weather_data):,} rows to {output_path}")
    return output_path


def download_live_forecast(forecast_days=7, session=None, models=PINNED_PREVIOUS_RUNS_MODEL):
    """Save a snapshot of the live forecast, with when it was taken.

    Each run writes its own timestamped file, so the forecasts that were actually
    available at prediction time can be scored later. Taken at 08:30 on the day
    before, the target day is about 15-39 hours ahead: fresher than the 48-hour
    forecast used in validation.
    """
    session = session or create_cached_session(expire_after=0)
    weather_data = fetch_live_forecast(forecast_days=forecast_days, session=session, models=models)
    weather_data["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")
    validate_weather_data(weather_data)

    output_path = get_output_file_path("live_forecast")
    save_weather_data(weather_data, output_path)
    print(f"Saved {len(weather_data):,} rows to {output_path}")
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

    previous_runs = subparsers.add_parser(
        "previous-runs",
        help="Download genuine past forecasts at fixed lead times (1-7 days ahead).",
    )
    previous_runs.add_argument("--start-date", required=True)
    previous_runs.add_argument("--end-date", required=True)
    previous_runs.add_argument(
        "--lead-days",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREVIOUS_RUNS_LEAD_DAYS),
        help="Forecast lead times in days, e.g. --lead-days 1 2.",
    )
    previous_runs.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_CHUNK_DELAY_SECONDS,
        help="Seconds to wait between chunks.",
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

    if args.command == "previous-runs":
        return download_previous_runs(
            start_date=args.start_date,
            end_date=args.end_date,
            lead_days=tuple(args.lead_days),
            sleep_seconds=args.sleep_seconds,
        )

    if args.command == "live-forecast":
        return download_live_forecast(forecast_days=args.forecast_days)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
