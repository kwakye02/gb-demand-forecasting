# GB Electricity Demand Forecasting

A day-ahead forecast of Great Britain's national electricity demand, for each half-hour of the day,
compared with the forecast NESO/Elexon publish for the same day.

Every forecast is made at **08:30 UK time on the day before**. Nothing published after that is
used, in training or in the live run.

## Results

Mean absolute error on National Demand (`ND`), in MW, over every half-hour of the year. The model
is trained only on days before the year it is scored on.

| Year | Seasonal naive | This model | NESO/Elexon day-ahead |
|---|---|---|---|
| 2025 | 2,177 | **758** | 711 |
| 2026 (to early Nov) | 2,111 | **825** | 789 |

- The model cuts the seasonal-naive error by about 60-65%.
- NESO's forecast is still 5-7% better. It is the newest version published by 09:00, half an hour
  after the model's cutoff, and NESO sees data from the whole system.
- The weather in these scores is the 48-hour-ahead ECMWF forecast, not the weather that actually
  happened, so weather-forecast error is part of the result.

## How it works

```
Elexon demand ─┐
ERA5 weather ──┤
Forecast wx ───┼─► gb_demand_merged.parquet ─► lag + weather features ─► CatBoost + Ridge ─► MLflow registry
NESO embedded ─┤                                                                               │
Elexon NDF ────┘  (benchmark only)                                       live day-ahead forecast ◄┘
```

**Features.** Everything is known by 08:30 on the day before:

- Demand lags of at least two days: the same time 2 and 7 days earlier, weekly lags out to 42 days,
  and a six-week average that skips bank-holiday weeks.
- NESO's embedded wind and solar forecast (the newest version issued by 08:30), and how it differs
  from the same time a week earlier and from the six-week baseline.
- Forecast weather for the target day (temperature, wind, cloud, radiation, humidity), corrected for
  its recent bias using only data already known (60-day window, 7-day delay).
- Calendar: day of week, month, bank holidays, the day before a holiday, settlement period.

**Model.** A blend of 75% CatBoost and 25% Ridge regression, with the Ridge fitted on the deviation
from the six-week baseline. Recent days count more: a day's training weight halves every year. The
model is trained on actual weather and predicts with corrected forecast weather.

**Live forecast.** `predict.py` builds the same feature matrix from live inputs and loads the
champion model, which is stored in the repo under `models/champion/`. A replay test checks that the
live-built matrix equals the training-path matrix on past days, including both clock changes
(46- and 50-period days).

**Live scoring.** Once a day is complete, `score.py` compares each saved forecast with actual demand
and with the NESO benchmark. It writes one row per day to `outputs/predictions/scores.parquet`: MAE
and bias for the model and for NESO, plus the model version.

## Quick start

```bash
git clone https://github.com/kwakye02/gb-demand-forecasting.git
cd gb-demand-forecasting
conda env create -f environment.yml && conda activate gb-demand-forecasting
```

```bash
python main.py                                  # download and build data/processed/ (first run is slow)
python -m gb_demand_forecasting.models.train    # fit the model and register it in MLflow
python -m gb_demand_forecasting.predictions.predict            # forecast tomorrow
python -m gb_demand_forecasting.predictions.predict --date 2026-11-12 --no-save
python -m gb_demand_forecasting.predictions.score              # score forecasts whose day is complete
```

Forecasts are appended to `outputs/predictions/forecasts.parquet`, one row per half-hour. After
training, set the `champion` alias on the version you want to serve in the local MLflow registry
(`mlflow.db`), then copy that version's model and `model_info.json` into `models/champion/`.

Other `main.py` options:

```bash
python main.py --end-date 2026-10-01   # stop at a fixed date
python main.py --skip-download         # only rebuild the processed files
```

## Daily run in the cloud

Two GitHub Actions workflows run the pipeline on a clean machine, so nothing depends on a laptop
being on. Both are started by hand for now, and the daily schedule is added once they run cleanly.

| Workflow | What it does |
|---|---|
| `data-refresh` | Restores the raw data from S3, downloads what is new, rebuilds `data/processed/`, saves both back |
| `daily-forecast` | Does the same refresh, scores finished forecast days, then forecasts tomorrow, saving each result to S3 |

State is kept in a private S3 bucket (`raw/`, `processed/`, `forecasts/`, `scores/`), and the
workflows use temporary AWS access, so no keys are stored in GitHub. With the raw data in S3, a
refresh takes about 2 minutes instead of 39.

## Data

| Source | What | Used for |
|---|---|---|
| [Elexon Insights](https://bmrs.elexon.co.uk/) INDO | Half-hourly national demand (`ND`), from 2020 | target |
| Elexon NDF | Every published NESO day-ahead forecast version, from mid-2021 | benchmark |
| [NESO Data Portal](https://www.neso.energy/data-portal) | Embedded wind and solar forecast archive | feature |
| [Open-Meteo](https://open-meteo.com/) ERA5 | Actual weather, 9 GB cities | training |
| Open-Meteo previous runs | 48-hour-ahead ECMWF IFS forecast weather, from 2024-07 | scoring and live |

Worth knowing:

- **Gaps.** Elexon had 414 half-hours gaps on 20 days in 2020-2023. They are filled with
  NESO's published `ND` for the same period (`reference/neso_demand_for_elexon_gaps.csv`) and
  flagged `is_imputed`. Nothing is missing from 2024 onwards.
- **Benchmark timing.** NESO publishes its full-day forecast at about 08:45, after the 08:30
  prediction time, so the benchmark is the newest version published by 09:00.
- **Forecast-weather gap.** ECMWF cloud cover is missing from 17 to 22 April 2026 in the Open-Meteo.
- **Weather lag.** ERA5 runs about 5 days behind, so training data ends about 5 days before today.
  The live run doesn't need it, since it uses forecast weather and demand up to two days back.
- **Embedded forecast timing.** NESO re-issues the forecast hourly, and an issue becomes visible
  about 28 minutes after its timestamp. The 08:00 issue is usable from about 08:30, so run the daily
  job at around 08:45.

Data is used under each provider's own terms: Elexon and NESO open data licences, and Open-Meteo
(CC BY 4.0, built on ECMWF and Copernicus ERA5). Please credit them if you reuse the outputs.

## Project layout

```
main.py                        download, process and build the merged dataset
gb_demand_forecasting/
  data/
    demand/                    Elexon downloader and loader (INDO demand, day-ahead benchmark)
    weather/                   Open-Meteo downloader (api.py) and the two weather loaders
    embedded_forecast.py       NESO embedded wind and solar forecast
    download_utils.py          shared download helpers and the 08:30 / 09:00 cutoff times
    build_dataset.py           merges everything into data/processed/gb_demand_merged.parquet
  features/                    lags.py, weather.py (bias correction), model_inputs.py
  models/                      catboost_ridge_blend.py, train.py (logs and registers in MLflow)
  predictions/                 target_periods.py, predict.py (live forecast), score.py (live scoring)
models/champion/               the model predict.py serves, with its feature list
.github/workflows/             the cloud data-refresh and daily-forecast workflows
notebooks/                     01-05: EDA, features, benchmarks, model selection
reference/                     small tracked reference data (NESO demand for Elexon's gaps)
tests/                         pytest suite
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests
```

## Status

Working: data pipeline, features, model, MLflow registry, live forecast, replay test, live scoring
against actual demand and the NESO forecast, and a cloud data refresh that rebuilds the dataset on a
clean runner.

In progress: the daily cloud forecast.

Next: turn on the daily schedule, add periodic retraining, add a rolling MAE to the scores, and
build a small dashboard.
