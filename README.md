# GB Electricity Demand Forecasting

A day-ahead forecast of Great Britain's national electricity demand (half-hourly, 48 periods a
day). Benchmarked against the forecast that NESO/Elexon publish for the same day.

Every forecast is made at **08:30 UK time on the day before**. Nothing published after that
moment is used, in training or in the live run.

## Results

Mean absolute error on National Demand (`ND`), in MW, over all half-hours of each year.
The model is trained only on days before the year it is scored on.

| Year | Seasonal naive | This model | NESO/Elexon day-ahead |
|---|---|---|---|
| 2025 | 2,177 | **758** | 711 |
| 2026 (to early Nov) | 2,111 | **825** | 789 |

- The model cuts the error of a seasonal-naive baseline by roughly 60-65%.
- It does **not** beat the official forecast: it is about 5-7% worse. The benchmark is the newest
  NESO forecast published by 09:00, so it sees 30 minutes more information than the model, and NESO
  has data from the whole system. A model within a few percent of it, from public data only, is
  the realistic goal.
- Weather at forecast time is the 48-hour-ahead ECMWF forecast (not the actual weather), with a
  rolling bias correction, so the scores include forecast-weather error.

## How it works

```
Elexon demand ─┐
ERA5 weather ──┤
Forecast wx ───┼─► gb_demand_merged.parquet ─► lag + weather features ─► CatBoost + Ridge ─► MLflow registry
NESO embedded ─┤                                                                               │
Elexon NDF ────┘  (benchmark only)                                       live day-ahead forecast ◄┘
```

**Features** (all computable at 08:30 on D-1, which is the only rule):

- Demand lags of at least two days: same time 2 and 7 days earlier, weekly lags out to 42 days,
  and a six-week average that leaves out bank-holiday weeks.
- NESO's embedded wind and solar forecast (newest version issued by 08:30), and how it differs
  from the same time a week earlier and from the six-week baseline.
- Forecast weather for the target day (temperature, wind, cloud, radiation, humidity), corrected
  for its recent bias using only data that was already known (60-day window, 7-day delay).
- Calendar: day of week, month, bank holidays, the day before a holiday, settlement period.

**Model.** A blend of 75% CatBoost and 25% Ridge regression (the Ridge fitted on the deviation from
the six-week baseline). Recent days count more: a day's training weight halves every year. It is
trained on actual weather and predicts with corrected forecast weather.

**Live forecast.** `predict.py` rebuilds the same feature matrix from live inputs and loads the
`champion` model from the MLflow registry. A replay test checks that the live-built matrix equals
the training-path matrix on a set of past days, including both clock changes (46- and 50-period
days).

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
```

Forecasts are appended to `outputs/predictions/forecasts.parquet` (one row per half-hour, with the
embedded-forecast issue time and model version). The model registry is a local SQLite file
(`mlflow.db`); after training, set the `champion` alias on the version you want to serve.

Other `main.py` options:

```bash
python main.py --end-date 2026-10-01   # stop at a fixed date
python main.py --skip-download         # only rebuild the processed files
```

## Data

| Source | What | Used for |
|---|---|---|
| [Elexon Insights](https://bmrs.elexon.co.uk/) INDO | Half-hourly national demand (`ND`), from 2020 | target |
| Elexon NDF | Every published NESO day-ahead forecast version, from mid-2021 | benchmark |
| [NESO Data Portal](https://www.neso.energy/data-portal) | Embedded wind and solar forecast archive | feature |
| [Open-Meteo](https://open-meteo.com/) ERA5 | Actual weather, 9 GB cities | training |
| Open-Meteo previous runs | 48-hour-ahead ECMWF IFS forecast weather, from 2024-07 | scoring and live |

Things worth knowing:

- **Gaps.** Elexon never published 414 half-hours on 20 days in 2020-2023. They are filled from
  NESO's published `ND` for the same period (`reference/neso_demand_for_elexon_gaps.csv`) and flagged
  `is_imputed`. Nothing is missing from 2024 onwards.
- **Benchmark timing.** NESO publishes its full-day forecast at about 08:45, after the 08:30
  prediction time, so the benchmark uses the newest version published by 09:00. The cutoffs are
  `PREDICTION_TIME` and `BENCHMARK_CUTOFF_TIME` in `data/download_utils.py`.
- **Forecast-weather gap.** ECMWF cloud cover is missing from 17 to 22 April 2026 in the
  Open-Meteo Previous Runs archive. Exclude those dates when cloud cover is used as a feature.
- **Weather lag.** ERA5 runs about 5 days behind, so training data ends about 5 days before today.
  The live run does not need it: it uses forecast weather and demand up to two days back.
- **Embedded forecast timing.** NESO re-issues the forecast hourly, and an issue becomes visible
  about 28 minutes after its timestamp. The 08:00 issue is therefore usable from about 08:30,
  so schedule the daily run for around 08:45.

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
  predictions/                 target_periods.py, predict.py (live day-ahead forecast)
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

Working: data pipeline, features, model, MLflow registry, live forecast (first saved forecast:
2026-11-12), replay test.

Next: scheduled daily run and monthly retraining, a scoring job (live forecasts against actual
demand and the NESO forecast, rolling MAE), and a small dashboard.
