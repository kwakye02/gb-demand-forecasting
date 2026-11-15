import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from gb_demand_forecasting.predictions.predict import FORECASTS_PATH
from gb_demand_forecasting.data.build_dataset import DEMAND_PATH
from gb_demand_forecasting.data.demand.download_elexon_demand_data import build_day_ahead_benchmark


SCORES_PATH = FORECASTS_PATH.parent / "scores.parquet"

def score_predictions(forecast,actual, benchmark, target_date):
    forecast = forecast[forecast['target_date'] == target_date].copy()
    actual['SETTLEMENT_DATE'] = pd.to_datetime(actual['SETTLEMENT_DATE'])
    actual = actual[actual['SETTLEMENT_DATE'] == target_date].copy()

    benchmark = benchmark[benchmark['settlement_date'] == target_date].copy()

    combined = forecast.merge(actual, left_on=['settlement_period','target_date'],
                            right_on=['SETTLEMENT_PERIOD','SETTLEMENT_DATE'], how='inner')

    combined = combined.merge(benchmark, left_on=['settlement_period', 'target_date'],
                              right_on=['settlement_period', 'settlement_date'], how='inner')
    
    if len(combined) != len(forecast):
        raise ValueError(f"Forecast and actual data do not match for {target_date}. Forecast has {len(forecast)} rows, actual has {len(actual)} rows, combined has {len(combined)} rows.")
    combined = combined[['settlement_period','target_date','model_version','forecast_mw','ND','elexon_forecast']].copy() 
    combined['Error_mw'] = (combined['forecast_mw']- combined['ND'])
    combined['Benchmark_Error_mw'] = (combined['elexon_forecast']- combined['ND'])

    return combined

def summarize_predictions(results):
    mae = np.mean(np.abs(results['Error_mw']))
    Date = results['target_date'].iloc[0]
    model_version = results['model_version'].iloc[0]
    number_of_periods = len(results)
    bias = np.mean(results['Error_mw'])
    benchmark_bias = np.mean(results['Benchmark_Error_mw'])
    elexon_mae = np.mean(np.abs(results['Benchmark_Error_mw']))
    return pd.DataFrame({'Date':[Date],'Model_Version':[model_version],
                         'Number_of_Periods':[number_of_periods],
                         'MAE':[round(mae, 1)],'Bias':[round(bias, 1)],
                         'Elexon_MAE':[round(elexon_mae, 1)],
                         'Elexon_Bias':[round(benchmark_bias, 1)]})


def score_new_days(forecasts, actual, benchmark, scores_path=SCORES_PATH):
    """Score every forecast day that is not in the scores file yet, and save the result."""
    scores = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
    already_scored = set(scores["Date"]) if len(scores) else set()
    new_rows = []

    for day in sorted(set(forecasts["target_date"]) - already_scored):
        try:
            results = score_predictions(forecasts, actual, benchmark, day)
        except ValueError as error:
            print(f"Skipping {day.date()}: {error}")
            continue
        new_rows.append(summarize_predictions(results))

    if new_rows:
        scores = pd.concat([scores, *new_rows], ignore_index=True).sort_values("Date")
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path, index=False)

    return scores


if __name__ == "__main__":
    scores = score_new_days(
        pd.read_parquet(FORECASTS_PATH), pd.read_parquet(DEMAND_PATH), build_day_ahead_benchmark()
    )
    print(scores.to_string(index=False))