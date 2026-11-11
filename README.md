# GB Demand Forecasting

An end to end machine learning pipeline for forecasting Great Britain electricity demand.

Predictions are made at **08:30 UK time on the day before** the settlement date.

## Data

Run from the project root:

```bash
python main.py                         # refresh everything up to yesterday and rebuild
python main.py --end-date 2026-10-01   # stop at a fixed date
python main.py --skip-download         # only rebuild the processed files
python main.py --live                  # also save a timestamped live weather forecast snapshot
```

| Source | What | Where it lands |
|---|---|---|
| Elexon INDO | Half-hourly national demand (`ND`), from 2020 | `data/raw/elexon/national_demand/` |
| Elexon NDF | Every published day-ahead forecast version, from mid-2021 | `data/raw/elexon/day_ahead_forecast/` |
| Open-Meteo ERA5 | Actual weather, 9 GB cities | `data/raw/open_meteo/historical_weather/` |
| Open-Meteo previous runs | 48-hour-ahead forecast weather (ECMWF IFS, pinned), from 2024-07 | `data/raw/open_meteo/previous_runs/chunks_ecmwf_ifs025/` |

`python main.py` rebuilds `data/processed/`, ending in **`gb_demand_merged.parquet`**: the
dataset the notebooks start from (indexed by UTC `datetime`).

Things to know about the data:

- **Elexon gaps.** Elexon never published 414 half-hours on 20 days in 2020-2023. They were filled
  once from NESO `ND` plus the local Elexon-NESO offset (`reference/elexon_gap_fill.csv`) and are
  flagged `is_imputed`. Nothing is missing from 2024 onwards.
- **Benchmark.** `elexon_forecast` is the newest NESO day-ahead forecast published by 09:00 UK
  time on the day before. NESO publishes its full-day forecast at about 08:45, 15 minutes after
  the 08:30 prediction time, so no full-day forecast exists at 08:30 itself. The cutoffs are
  `PREDICTION_TIME` and `BENCHMARK_CUTOFF_TIME` in `gb_demand_forecasting/data/demand/elexon_demand.py`.
  Elexon has no forecast versions for May 2021 and almost none until July 2021.
- **Weather.** Training uses actual weather; evaluation uses the 48-hour forecast
  (`forecast_*` columns). ERA5 runs about 5 days behind, so the merged dataset ends about 5 days
  before today. ECMWF has no cloud cover for 2026-04-17 15:00 to 2026-04-22 23:00 UTC.
- **Live weather** snapshots (`--live`) are about 15-39 hours ahead of the target day, fresher
  than the 48-hour forecast used in validation.

## Layout

```
main.py                     one command: download, process, build
gb_demand_forecasting/
  data/demand/              Elexon downloader and demand loader
  data/weather/             Open-Meteo downloader and loaders
  data/build_dataset.py     merges everything into gb_demand_merged.parquet
  features/  models/        (for the modelling stage)
reference/                  small tracked reference data (the gap fill)
notebooks/                  01-05, one per stage; old copies in notebooks/archive/
archive/                    retired NESO code/data and legacy weather; see archive/MOVES.md
```

## Tests

```bash
python -m pytest tests
```
