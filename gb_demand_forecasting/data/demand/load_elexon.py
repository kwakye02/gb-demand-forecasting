"""Build the processed GB demand series from Elexon INDO, filling Elexon's known gaps."""

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ELEXON_DIR = PROJECT_ROOT / "data" / "raw" / "elexon" / "national_demand"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Elexon never published 414 periods on 20 days in 2020-2023. They were filled once from
# NESO ND plus the local Elexon-NESO offset; reference/elexon_gap_fill.csv records the result.
GAP_FILL_PATH = PROJECT_ROOT / "reference" / "elexon_gap_fill.csv"

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


def load_gap_fill(path=GAP_FILL_PATH):
    """Read the recorded fill values for Elexon's known gaps."""
    fill = pd.read_csv(path, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    return fill.rename(
        columns={
            "SETTLEMENT_DATE": "settlement_date",
            "SETTLEMENT_PERIOD": "settlement_period",
            "ND": "fill_mw",
        }
    ).assign(settlement_date=lambda x: pd.to_datetime(x["settlement_date"]))


def periods_in_day(day):
    """Number of half-hour settlement periods on a UK day (46, 48 or 50 on clock changes)."""
    start = pd.Timestamp(day).tz_localize("Europe/London")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/London")
    return int((end - start).total_seconds() // 1800)


def build_settlement_grid(first_day, last_day, last_period_on_last_day=None):
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


def fill_known_gaps(grid, elexon, gap_fill):
    """Fill the recorded gaps; any other missing period is left empty and reported."""
    df = grid.merge(elexon[[*KEY, "demand_mw"]], on=KEY, how="left").merge(
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
    grid = build_settlement_grid(
        elexon["settlement_date"].min(),
        latest["settlement_date"],
        last_period_on_last_day=int(latest["settlement_period"]),
    )

    outside_grid = len(
        elexon.merge(grid, on=KEY, how="left", indicator=True).query("_merge == 'left_only'")
    )

    if outside_grid:
        raise ValueError(
            f"{outside_grid} Elexon periods fall outside the expected settlement grid."
        )

    df = fill_known_gaps(grid, elexon, load_gap_fill())

    if df.duplicated(KEY).any():
        raise ValueError("Duplicate settlement periods in the demand series.")

    still_missing = df.loc[df["demand_mw"].isna(), "settlement_date"]

    if len(still_missing):
        print(
            f"WARNING: {len(still_missing)} periods are missing from Elexon and not in the "
            f"gap-fill reference (dates: {sorted(still_missing.dt.strftime('%Y-%m-%d').unique())[:10]})."
        )

    # Same names as the old NESO file, so the downstream notebooks keep working.
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
