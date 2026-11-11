"""NESO embedded wind and solar forecast: the newest version issued before the prediction time.

National Demand excludes embedded (rooftop and small-scale) generation, so a forecast of it is
the most useful extra input for a day-ahead demand model. NESO republishes this forecast hourly.
For every settlement half-hour this module keeps the newest version issued by PREDICTION_TIME
(08:30 UK) on the day before, which is what would have been known when the prediction was made.

The selection runs on the NESO data portal's own query API, so only the small result is
downloaded (about 1,500 rows a month) instead of the multi-gigabyte yearly archives.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path
import re

import pandas as pd

from gb_demand_forecasting.data.demand.elexon_demand import (
    DEFAULT_START_DATE,
    PREDICTION_TIME,
    build_session,
    download_monthly_chunks,
    read_chunks,
)

API_URL = "https://api.neso.energy/api/3/action"
DATASET_ID = "embedded-wind-and-solar-forecasts"
ARCHIVE_NAME = "Embedded Solar and Wind Forecast Archive {year}"
CURRENT_NAME = "Embedded Solar and Wind Forecast"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_NESO_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "neso"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHUNK_NAME = "embedded_forecast"
PROCESSED_NAME = "embedded_forecast_0830.parquet"

REQUEST_TIMEOUT = 170
KEY = ["settlement_date", "settlement_period"]
OUTPUT_COLUMNS = [*KEY, "emb_solar_fc", "emb_wind_fc", "emb_fc_issued_utc"]
RESOURCE_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def get_resource_ids(session):
    """Map each archive year to its resource id, plus the key 'current' for the rolling file."""

    response = session.get(
        f"{API_URL}/package_show", params={"id": DATASET_ID}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    resources = {item["name"]: item["id"] for item in response.json()["result"]["resources"]}

    ids = {
        year: resources[ARCHIVE_NAME.format(year=year)]
        for year in range(2019, date.today().year + 1)
        if ARCHIVE_NAME.format(year=year) in resources
    }
    ids["current"] = resources[CURRENT_NAME]
    return ids


def cutoff_offset(cutoff_time=PREDICTION_TIME, cutoff_days_before=1):
    """How long before UK midnight of the settlement date the forecast must have been issued."""

    clock = pd.to_timedelta(f"{cutoff_time}:00")
    offset = pd.Timedelta(days=cutoff_days_before) - clock
    hours, minutes = int(offset.total_seconds() // 3600), int(offset.total_seconds() % 3600 // 60)
    return f"{hours} hours {minutes} minutes"


def build_month_query(
    resource_id, start_date, end_date, cutoff_time=PREDICTION_TIME, cutoff_days_before=1
):
    """SQL that returns, per settlement half-hour, the newest forecast issued by the cutoff."""

    if not RESOURCE_ID.match(resource_id):
        raise ValueError(f"Invalid resource id: {resource_id!r}")

    start = date.fromisoformat(str(start_date))
    stop = date.fromisoformat(str(end_date)) + timedelta(days=1)
    interval = cutoff_offset(cutoff_time, cutoff_days_before)

    # The cutoff is a UK wall-clock time on the day before, so it is converted to UTC in
    # the database, which is what keeps it right on both sides of a clock change.
    # Three off-schedule versions on 2026-06-12 were stored with the half-hour's own time in
    # SETTLEMENT_DATE instead of midnight. The cutoff cannot be worked out for them, so rows
    # that are not at midnight are excluded.
    return f"""
SELECT DISTINCT ON ("SETTLEMENT_DATE", "SETTLEMENT_PERIOD")
       "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "EMBEDDED_SOLAR_FORECAST",
       "EMBEDDED_WIND_FORECAST", "Forecast_Datetime"
FROM "{resource_id}"
WHERE "SETTLEMENT_DATE" >= '{start.isoformat()}' AND "SETTLEMENT_DATE" < '{stop.isoformat()}'
  AND "SETTLEMENT_DATE" = date_trunc('day', "SETTLEMENT_DATE")
  AND "Forecast_Datetime" <=
      (("SETTLEMENT_DATE" - interval '{interval}') AT TIME ZONE 'Europe/London') AT TIME ZONE 'UTC'
ORDER BY "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "Forecast_Datetime" DESC
""".strip()


def normalise_records(records):
    """Convert API records into the standard columns and types."""

    columns = [*OUTPUT_COLUMNS]

    if not records:
        return pd.DataFrame(columns=columns)

    raw = pd.DataFrame(records)
    required = {
        "SETTLEMENT_DATE",
        "SETTLEMENT_PERIOD",
        "EMBEDDED_SOLAR_FORECAST",
        "EMBEDDED_WIND_FORECAST",
        "Forecast_Datetime",
    }
    missing = required - set(raw.columns)

    if missing:
        raise ValueError(f"Missing NESO response columns: {sorted(missing)}")

    not_midnight = ~raw["SETTLEMENT_DATE"].astype(str).str[10:].isin(
        ["", "T00:00:00", " 00:00:00"]
    )

    if not_midnight.any():
        raise ValueError(
            f"{int(not_midnight.sum())} rows have a SETTLEMENT_DATE that is not at midnight; "
            "the cutoff cannot be applied to them."
        )

    df = pd.DataFrame(
        {
            "settlement_date": pd.to_datetime(raw["SETTLEMENT_DATE"].astype(str).str[:10]),
            "settlement_period": pd.to_numeric(raw["SETTLEMENT_PERIOD"], errors="raise").astype(
                int
            ),
            "emb_solar_fc": pd.to_numeric(raw["EMBEDDED_SOLAR_FORECAST"], errors="raise"),
            "emb_wind_fc": pd.to_numeric(raw["EMBEDDED_WIND_FORECAST"], errors="raise"),
            "emb_fc_issued_utc": pd.to_datetime(raw["Forecast_Datetime"], utc=True),
        }
    )
    return df[columns]


def check_issued_by_cutoff(df, cutoff_time=PREDICTION_TIME, cutoff_days_before=1):
    """Raise if any forecast was issued after the cutoff, so nothing from the future gets in."""

    if df.empty:
        return

    cutoff_local = (
        df["settlement_date"]
        - pd.Timedelta(days=cutoff_days_before)
        + pd.to_timedelta(f"{cutoff_time}:00")
    )
    cutoff = cutoff_local.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    late = df["emb_fc_issued_utc"] > cutoff

    if late.any():
        first = df.loc[late, "settlement_date"].min().date()
        raise ValueError(
            f"{int(late.sum())} forecasts were issued after the {cutoff_time} cutoff "
            f"(first on settlement date {first})."
        )


def newest_per_period(df):
    """Keep the most recently issued forecast for each settlement half-hour."""

    return (
        df.sort_values("emb_fc_issued_utc")
        .groupby(KEY, as_index=False)
        .tail(1)
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def resources_to_query(resource_ids, year, month):
    """Archive resources holding the forecasts for a month.

    Forecasts for 1-13 January are issued in December, so January also needs the previous
    year's archive.
    """

    years = [year] + ([year - 1] if month == 1 else [])
    return [resource_ids[y] for y in years if y in resource_ids]


def fetch_embedded_forecast(
    start_date,
    end_date,
    session=None,
    resource_ids=None,
    cutoff_time=PREDICTION_TIME,
    cutoff_days_before=1,
):
    """
    Download the newest embedded forecast issued by the cutoff, for each half-hour in a month.

    Parameters
    ----------
    start_date, end_date:
        Settlement dates in YYYY-MM-DD format, inclusive, within one calendar month.

    Returns
    -------
    pd.DataFrame:
        settlement_date
        settlement_period
        emb_solar_fc
        emb_wind_fc
        emb_fc_issued_utc
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date must be before end_date")

    if (start.year, start.month) != (end.year, end.month):
        raise ValueError("fetch_embedded_forecast covers one calendar month at a time")

    own_session = session is None
    session = session or build_session()

    try:
        resource_ids = resource_ids or get_resource_ids(session)
        frames = []

        for resource_id in resources_to_query(resource_ids, start.year, start.month):
            sql = build_month_query(
                resource_id, start_date, end_date, cutoff_time, cutoff_days_before
            )
            response = session.get(
                f"{API_URL}/datastore_search_sql", params={"sql": sql}, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            payload = response.json()

            if not payload.get("success"):
                raise RuntimeError(f"NESO query failed: {str(payload.get('error'))[:300]}")

            frames.append(normalise_records(payload["result"]["records"]))
    finally:
        if own_session:
            session.close()

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = newest_per_period(pd.concat(frames, ignore_index=True))

    if df.duplicated(KEY).any():
        raise ValueError("Duplicate settlement periods in the embedded forecast")

    check_issued_by_cutoff(df, cutoff_time, cutoff_days_before)
    return df


def download_embedded_forecasts(
    start_date=DEFAULT_START_DATE,
    end_date=None,
    output_directory=RAW_NESO_DIRECTORY,
):
    """Download the 08:30 forecast to monthly chunks; the current month is refreshed each run."""

    end_date = end_date or (date.today() - timedelta(days=1)).isoformat()
    session = build_session()
    resource_ids = get_resource_ids(session)

    def fetch(chunk_start, chunk_end):
        return fetch_embedded_forecast(chunk_start, chunk_end, session, resource_ids)

    return download_monthly_chunks(
        CHUNK_NAME, fetch, start_date, end_date, output_directory, prefix="gb_neso"
    )


def build_embedded_forecast_file(output_directory=RAW_NESO_DIRECTORY, save=True):
    """Combine the monthly chunks into the processed file used by the dataset build."""

    df = read_chunks(Path(output_directory) / CHUNK_NAME / "chunks")
    df = newest_per_period(df)

    if df.duplicated(KEY).any():
        raise ValueError("Duplicate settlement periods across chunks")

    check_issued_by_cutoff(df)

    df = df.rename(
        columns={
            "settlement_date": "SETTLEMENT_DATE",
            "settlement_period": "SETTLEMENT_PERIOD",
        }
    )
    df["SETTLEMENT_DATE"] = df["SETTLEMENT_DATE"].dt.strftime("%Y-%m-%d")

    print(
        f"Embedded forecast: {df.shape} | {df['SETTLEMENT_DATE'].min()} to {df['SETTLEMENT_DATE'].max()}"
    )

    if save:
        out_path = PROCESSED_DATA_DIR / PROCESSED_NAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    return df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the NESO embedded wind and solar forecast (newest issued by 08:30 UK on D-1)."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end-date", default=None, help="Last date (YYYY-MM-DD). Defaults to yesterday."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    download_embedded_forecasts(args.start_date, args.end_date)
    build_embedded_forecast_file()


if __name__ == "__main__":
    main()
