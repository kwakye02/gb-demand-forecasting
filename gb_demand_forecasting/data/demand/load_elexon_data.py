"""Build the processed GB demand series from Elexon INDO and fill missing gaps with NESO ND."""

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ELEXON_DIR = PROJECT_ROOT / "data" / "raw" / "elexon" / "national_demand"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

NESO_DATA_PATH = PROJECT_ROOT / "reference" / "neso_demand_for_elexon_gaps.csv" #fill missing data using NESO ND

KEY = ["settlement_date", "settlement_period"]


def load_elexon_demand():
    """Read the combined Elexon INDO file."""
    files = glob.glob(str(RAW_ELEXON_DIR / "*.parquet"))

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one combined Elexon file in {RAW_ELEXON_DIR}, found {len(files)}."
        )

    df = pd.read_parquet(files[0], columns=[*KEY, "start_time_utc", "demand_mw"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df

def load_missing_demand(path=NESO_DATA_PATH):
    """Read NESO's published demand for the periods Elexon never published."""

    fill = pd.read_csv(path, usecols=["settlement_date", "settlement_period", "neso_nd_mw"])
    return fill.rename(columns={
        "neso_nd_mw": "fill_mw",
    }).assign(settlement_date=lambda x: pd.to_datetime(x["settlement_date"]))

def periods_in_day(day):
    """Number of half-hour settlement periods on a UK day (46, 48 or 50 on clock changes)."""
    start = pd.Timestamp(day).tz_localize("Europe/London")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/London")
    return int((end - start).total_seconds() // 1800)


def build_expected_periods(first_day, last_day, last_period_on_last_day=None):
    """Every settlement period that should exist between two dates.

    The final day stops at last_period_on_last_day, so periods Elexon has not
    published yet are not counted as gaps.
    """
    rows = []

    for day in pd.date_range(first_day, last_day):
        last_period = periods_in_day(day)

        if day == pd.Timestamp(last_day) and last_period_on_last_day is not None:
            last_period = min(last_period, last_period_on_last_day)

        rows.extend((day, period) for period in range(1, last_period + 1))

    return pd.DataFrame(rows, columns=KEY)



def fill_known_gaps(expected_periods, elexon, gap_fill):
    """Fill the recorded gaps; any other missing period is left empty and reported."""
    df = expected_periods.merge(elexon[[*KEY, "demand_mw"]], on=KEY, how="left").merge(
        gap_fill, on=KEY, how="left"
    )
    fill_rows = df["demand_mw"].isna() & df["fill_mw"].notna()
    df.loc[fill_rows, "demand_mw"] = df.loc[fill_rows, "fill_mw"]
    df["is_imputed"] = fill_rows
    return df.drop(columns="fill_mw")


def run_elexon_pipeline(save=True):
    """Build the complete half-hourly demand series from Elexon."""
    elexon = load_elexon_demand()
    latest = elexon.sort_values("start_time_utc").iloc[-1]
    expected_periods = build_expected_periods(
        elexon["settlement_date"].min(),
        latest["settlement_date"],
        last_period_on_last_day=int(latest["settlement_period"]),
    )

    n_unexpected_periods = len(
        elexon.merge(expected_periods, on=KEY, how="left", indicator=True).query("_merge == 'left_only'")
    )

    if n_unexpected_periods:
        raise ValueError(
            f"{n_unexpected_periods} Elexon periods fall outside the expected settlement periods."
        )

    df = fill_known_gaps(expected_periods, elexon, load_missing_demand())

    if df.duplicated(KEY).any():
        raise ValueError("Duplicate settlement periods in the demand series.")

    still_missing = df.loc[df["demand_mw"].isna(), "settlement_date"]

    if len(still_missing):
        print(
            f"WARNING: {len(still_missing)} periods are missing from Elexon and not in the "
            f"gap-fill reference (dates: {sorted(still_missing.dt.strftime('%Y-%m-%d').unique())[:10]})."
        )

    #rename columns to match the rest of the pipeline and save.
    df = df.rename(
        columns={
            "settlement_date": "SETTLEMENT_DATE",
            "settlement_period": "SETTLEMENT_PERIOD",
            "demand_mw": "ND",
        }
    )
    df["SETTLEMENT_DATE"] = df["SETTLEMENT_DATE"].dt.strftime("%Y-%m-%d")

    print(f"Final shape: {df.shape}")
    print(f"Imputed periods: {int(df['is_imputed'].sum())}")

    if save:
        out_path = PROCESSED_DATA_DIR / "gb_demand_elexon.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path}")

    return df


if __name__ == "__main__":
    run_elexon_pipeline()
