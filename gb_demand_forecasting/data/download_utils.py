"""
Shared utilities for downloading and managing energy data.

Includes HTTP session configuration, monthly data downloads,
file caching, and forecasting cutoff times.
"""

from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_START_DATE = "2020-01-01"

# UK-local information cutoffs on the day before the target date.
PREDICTION_TIME = "08:30"
BENCHMARK_CUTOFF_TIME = "09:00"

def build_session():
    """Create a reusable HTTP session with automatic retries."""

    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524], 
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    session.mount("https://",HTTPAdapter(max_retries=retries))

    session.headers.update(
        {
            "User-Agent": "gb-demand-forecasting/1.0",
            "Accept": "application/json",
        }
    )

    return session


def get_monthly_date_range(start_date, end_date):
    """Split a date range into month_start and month_end dates."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    chunks = []

    for month in pd.period_range(start, end, freq="M"):
        chunks.append(
            (
                max(month.start_time.normalize(), start).strftime("%Y-%m-%d"),
                min(month.end_time.normalize(), end).strftime("%Y-%m-%d"),
            )
        )

    return chunks


def read_chunks(chunk_directory):
    """Concatenate every saved monthly chunk in a directory."""

    files = sorted(Path(chunk_directory).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No chunks found in {chunk_directory}")

    return pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True,
    )


def download_and_save_monthly_data(
    name, fetch, start_date, end_date, output_directory, prefix="gb_elexon"
):
    """
    Download and save data in monthly Parquet files.

    Reuse previously downloaded complete months and refresh
    the final requested month.
    """

    chunk_directory = Path(output_directory) / name / "chunks"
    chunk_directory.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(end_date)

    for chunk_start, chunk_end in get_monthly_date_range(start_date, end_date):
        month = chunk_start[:7]
        chunk_path = chunk_directory / f"{prefix}_{name}_{month}.parquet"
        month_is_complete = (
            pd.Timestamp(chunk_start).day == 1
            and pd.Timestamp(chunk_end) == pd.Timestamp(chunk_end) + pd.offsets.MonthEnd(0)
            and pd.Timestamp(chunk_end) < end
        )

        if chunk_path.exists() and month_is_complete:
            print(f"Using cached chunk {chunk_path.name}")
            continue

        try:
            chunk = fetch(chunk_start, chunk_end)
        except ValueError as error:
            print(f"{month}: nothing downloaded ({error})")
            continue

        if chunk.empty:
            print(f"{month}: no data published")
            continue

        chunk.to_parquet(chunk_path, index=False)
        print(f"Saved {len(chunk):,} rows to {chunk_path.name}")

    return chunk_directory
