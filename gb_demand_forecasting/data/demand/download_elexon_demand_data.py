"""Download half-hourly GB national demand and day-ahead demand forecasts from Elexon BMRS."""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gb_demand_forecasting.data.download_utils import (
    BENCHMARK_CUTOFF_TIME,
    DEFAULT_START_DATE,
    build_session,
    download_and_save_monthly_data,
    read_chunks,
)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ELEXON_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "elexon"

DAY_AHEAD_FORECAST_FIRST_DATE = "2021-04-01"  # First date for which Elexon provides day-ahead forecasts


def json_to_df(response_data):
    """Convert an Elexon JSON response into a DataFrame."""

    if isinstance(response_data, list):
        return pd.DataFrame(response_data)

    if isinstance(response_data, dict):
        if isinstance(response_data.get("data"), list):
            return pd.DataFrame(response_data["data"])

    raise ValueError("Unexpected Elexon API response format")


def fetch_df(session, endpoint, params=None):
    """Request data from an Elexon endpoint."""

    url = f"{BASE_URL}/{endpoint.lstrip('/')}"

    response = session.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return json_to_df(response.json())


def fetch_national_demand(start_date,end_date,as_of=None):
    """
    Download Elexon Initial National Demand Outturn (INDO).

    Parameters
    ----------
    start_date, end_date:
        Settlement dates in YYYY-MM-DD format, inclusive.

    as_of:
        Optional timezone-aware datetime.
        Only keep observations published by this cutoff.

    Returns
    -------
    pd.DataFrame:
        settlement_date
        settlement_period
        start_time_utc
        publish_time_utc
        demand_mw
    """

    start = date.fromisoformat(str(start_date))
    end = date.fromisoformat(str(end_date))

    if start > end:
        raise ValueError("start_date must be before end_date")

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)

        if cutoff.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")

        cutoff = cutoff.tz_convert("UTC")
    else:
        cutoff = None

    parts = []

    with build_session() as session:
        # Downloads the data in small chunks rather than one large request.
        chunk_start = start

        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=6),end)

            params = {
                "settlementDateFrom": chunk_start.isoformat(),
                "settlementDateTo": chunk_end.isoformat(),
            }

            df = fetch_df(session,"/demand/outturn/stream", params=params)

            if not df.empty:
                parts.append(df)

            chunk_start = chunk_end + timedelta(days=1)

    if not parts:
        raise ValueError("Elexon returned no demand observations")

    df = pd.concat(parts, ignore_index=True)

    required_columns = [
        "settlementDate",
        "settlementPeriod",
        "startTime",
        "publishTime",
        "initialDemandOutturn",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing Elexon response columns: {sorted(missing)}")

    df = df[required_columns].rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "startTime": "start_time_utc",
            "publishTime": "publish_time_utc",
            "initialDemandOutturn": "demand_mw",
        }
    )

    # Convert data types.
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

    df["settlement_period"] = pd.to_numeric(df["settlement_period"],errors="raise",).astype(int)

    for column in ["start_time_utc", "publish_time_utc"]:
        df[column] = pd.to_datetime(df[column],utc=True,errors="coerce")

    df["demand_mw"] = pd.to_numeric(df["demand_mw"],errors="coerce")

    # Reject records with an invalid settlement identity.
    if df["start_time_utc"].isna().any():
        raise ValueError("Invalid settlement start timestamps")

    if df.duplicated(["settlement_date", "settlement_period"]).any():
        raise ValueError("Duplicate settlement periods detected")

    # Prevent using information published after the cutoff.
    if cutoff is not None:
        df = df.loc[
            df["publish_time_utc"].notna()
            & (df["publish_time_utc"] <= cutoff)
            & (df["start_time_utc"] < cutoff)
        ].copy()

    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    return df


def fetch_day_ahead_forecasts(start_date, end_date):
    """
    Download every published Elexon National Demand Forecast (NDF) version.

    Each version is the day-ahead forecast as NESO published it at
    publish_time_utc, so the data can be cut to what was known at any time.

    Parameters
    ----------
    start_date, end_date:
        Publish dates in YYYY-MM-DD format, inclusive.

    Returns
    -------
    pd.DataFrame:
        settlement_date
        settlement_period
        start_time_utc
        publish_time_utc
        boundary
        forecast_mw

    Empty if nothing was published in the period (Elexon holds NDF from July 2021).
    """

    start = date.fromisoformat(str(start_date))
    end = date.fromisoformat(str(end_date))

    if start > end:
        raise ValueError("start_date must be before end_date")

    params = {
        "publishDateTimeFrom": f"{start.isoformat()}T00:00:00Z",
        "publishDateTimeTo": f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z",
    }

    with build_session() as session:
        df = fetch_df(session, "/datasets/NDF/stream", params=params)

    columns = [
        "settlement_date",
        "settlement_period",
        "start_time_utc",
        "publish_time_utc",
        "boundary",
        "forecast_mw",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    required_columns = [
        "settlementDate",
        "settlementPeriod",
        "startTime",
        "publishTime",
        "boundary",
        "demand",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing Elexon response columns: {sorted(missing)}")

    df = df[required_columns].rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "startTime": "start_time_utc",
            "publishTime": "publish_time_utc",
            "demand": "forecast_mw",
        }
    )

    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date

    df["settlement_period"] = pd.to_numeric(
        df["settlement_period"],
        errors="raise",
    ).astype(int)

    for column in ["start_time_utc", "publish_time_utc"]:
        df[column] = pd.to_datetime(
            df[column],
            utc=True,
            errors="coerce",
        )

    df["forecast_mw"] = pd.to_numeric(df["forecast_mw"],errors="coerce")

    if df[["start_time_utc", "publish_time_utc"]].isna().any().any():
        raise ValueError("Invalid forecast timestamps")

    df = df.drop_duplicates()

    if df.duplicated(
        ["publish_time_utc", "settlement_date", "settlement_period", "boundary"]
    ).any():
        raise ValueError("Conflicting duplicate forecast values detected")

    return df.sort_values(
        ["publish_time_utc", "settlement_date", "settlement_period"]
    ).reset_index(drop=True)


def latest_forecast_before_cutoff(
    forecasts,
    cutoff_time=BENCHMARK_CUTOFF_TIME,
    cutoff_days_before=1,
    local_timezone="Europe/London",
):
    """
    Keep, for each settlement period, the newest forecast published by the cutoff.

    The cutoff is cutoff_time local time, cutoff_days_before days before the
    settlement date (default: 09:00 the day before).
    Forecasts published after the benchmark cutoff are excluded.
    """

    forecasts = forecasts.copy()

    cutoff_local = (
        pd.to_datetime(forecasts["settlement_date"])
        - pd.Timedelta(days=cutoff_days_before)
        + pd.to_timedelta(f"{cutoff_time}:00")
    )

    forecasts["cutoff_utc"] = cutoff_local.dt.tz_localize(
        local_timezone, ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")

    eligible = forecasts.loc[
        forecasts["cutoff_utc"].notna()
        & (forecasts["publish_time_utc"] <= forecasts["cutoff_utc"])
    ]

    return (
        eligible.sort_values("publish_time_utc")
        .groupby(["settlement_date", "settlement_period", "boundary"], as_index=False)
        .tail(1)
        .sort_values(["settlement_date", "settlement_period"])
        .drop(columns="cutoff_utc")
        .reset_index(drop=True)
    )


def download_national_demand(
    start_date=DEFAULT_START_DATE,
    end_date=None,
    output_directory=RAW_ELEXON_DIRECTORY,
):
    """Download INDO to monthly chunks and a combined parquet file."""

    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    chunk_directory = download_and_save_monthly_data(
        "national_demand", fetch_national_demand, start_date, end_date, output_directory
    )

    demand = (
        read_chunks(chunk_directory)
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    if demand.duplicated(["settlement_date", "settlement_period"]).any():
        raise ValueError("Duplicate settlement periods across chunks")

    output_path = (
        Path(output_directory)
        / "national_demand"
        / (
            f"gb_elexon_national_demand_{demand['settlement_date'].min()}_"
            f"{demand['settlement_date'].max()}.parquet"
        )
    )

    for stale in output_path.parent.glob("gb_elexon_national_demand_*.parquet"):
        stale.unlink()

    demand.to_parquet(output_path, index=False)
    print(f"Saved {len(demand):,} rows to {output_path}")
    return output_path


def download_day_ahead_forecasts(
    start_date=DEFAULT_START_DATE,
    end_date=None,
    output_directory=RAW_ELEXON_DIRECTORY,
):
    """Download every NDF version to monthly chunks (no combined file; they are large)."""

    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    start_date = max(str(start_date), DAY_AHEAD_FORECAST_FIRST_DATE)
    return download_and_save_monthly_data(
        "day_ahead_forecast", fetch_day_ahead_forecasts, start_date, end_date, output_directory
    )


def build_day_ahead_benchmark(
    cutoff_time=BENCHMARK_CUTOFF_TIME,
    cutoff_days_before=1,
    output_directory=RAW_ELEXON_DIRECTORY,
):
    """
    Day-ahead benchmark: the newest NESO forecast published by the benchmark cutoff.

    Returns one row per settlement period with the forecast, and when it was published.
    """

    forecasts = read_chunks(Path(output_directory) / "day_ahead_forecast" / "chunks")
    selected = latest_forecast_before_cutoff(
        forecasts, cutoff_time=cutoff_time, cutoff_days_before=cutoff_days_before
    )

    return (
        selected[["settlement_date", "settlement_period", "publish_time_utc", "forecast_mw"]]
        .rename(
            columns={
                "publish_time_utc": "elexon_forecast_published_utc",
                "forecast_mw": "elexon_forecast",
            }
        )
        .assign(settlement_date=lambda x: pd.to_datetime(x["settlement_date"]))
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download Elexon national demand and day-ahead demand forecasts."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument( "--end-date",default=None,
        help="Last date to download (YYYY-MM-DD). Defaults to yesterday.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("\n=== Elexon national demand (INDO) ===")
    download_national_demand(args.start_date, args.end_date)

    print("\n=== Elexon day-ahead demand forecast versions (NDF) ===")
    download_day_ahead_forecasts(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
