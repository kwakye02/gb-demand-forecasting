"""Build the small JSON file that the dashboard reads.

The file has four sections: the latest forecast, the daily scores, a half-hour comparison of
forecast, actual and NESO demand for recent days, and the pipeline status.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from gb_demand_forecasting.data.build_dataset import DEMAND_PATH
from gb_demand_forecasting.data.demand.download_elexon_demand_data import build_day_ahead_benchmark
from gb_demand_forecasting.predictions.predict import FORECASTS_PATH
from gb_demand_forecasting.predictions.score import SCORES_PATH, score_predictions

SCHEMA_VERSION = 1
DEFAULT_REVIEW_DAYS = 30
DEFAULT_OUTPUT_PATH = FORECASTS_PATH.parent.parent / "dashboard" / "dashboard.json"


def get_latest_forecast(forecasts_path):
    """
    Return the newest saved day-ahead 
    forecast as a dictionary that can be written to JSON.
    """
    forecasts = pd.read_parquet(forecasts_path)
    if forecasts.empty:
        raise ValueError(f"No forecasts found in {forecasts_path}")

    last_date = forecasts["target_date"].max()
    df = forecasts[forecasts["target_date"] == last_date]

    # keep the newest forecast for each period.
    df = (df.sort_values("issued_at_utc")
              .drop_duplicates("settlement_period", keep="last")
              .sort_values("settlement_period")
    )


    peak = df.loc[df["forecast_mw"].idxmax()]
    energy_gwh = df["forecast_mw"].sum() * 0.5/ 1000

    results = {
        "target_date": pd.Timestamp(last_date).strftime("%Y-%m-%d"),
        "model_version": int(df.loc[df["issued_at_utc"].idxmax(), "model_version"]),
        "peak_mw": round(float(peak["forecast_mw"]), 1),
        "peak_period": int(peak["settlement_period"]),
        "energy_gwh": round(float(energy_gwh), 1),
        "periods": [
            {"settlement_period": int(period), "forecast_mw": round(float(demand), 1)}
            for period, demand in zip(df["settlement_period"], df["forecast_mw"])
        ]
    }
    return results


def get_daily_scores(scores_path):
    """
    Return one dictionary per scored day.
    
    """
    scores = pd.read_parquet(scores_path).sort_values("Date")
    
    results =[
        {
            "date": pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
            "model_version": int(row.Model_Version),
            "n_periods": int(row.Number_of_Periods),
            "mae": float(row.MAE),
            "bias": float(row.Bias),
            "neso_mae": float(row.Elexon_MAE),
            "neso_bias": float(row.Elexon_Bias),
        }
        for row in scores.itertuples()
    ]
    return results


def build_day_review(forecasts, scores, actual, benchmark, days=DEFAULT_REVIEW_DAYS):
    """
    Return forecast, actual and NESO demand for each
    half-hour of the latest scored days.
    """
    review = []
    for day in sorted(scores["Date"])[-days:]:
        # score_predictions adds columns to its inputs, so give it copies.
        try:
            day_results = score_predictions(
                forecasts.copy(), actual.copy(), benchmark.copy(), day)
        except ValueError:
            continue
        for row in day_results.itertuples():
            review.append(
                {
                    "date": pd.Timestamp(row.target_date).strftime("%Y-%m-%d"),
                    "settlement_period": int(row.settlement_period),
                    "forecast_mw": round(float(row.forecast_mw), 1),
                    "actual_mw": round(float(row.ND), 1),
                    "neso_mw": round(float(row.elexon_forecast), 1),
                }
            )
    return review


def check_pipeline_status(forecasts, scores, generated_at):
    """
    Return information about the model pipeline,
    including the last forecast issued, 
    the last scored day, and the model version.
    """
    status = forecasts.loc[forecasts["issued_at_utc"].idxmax()]
    results = {
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_forecast_issued_utc": pd.Timestamp(status["issued_at_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_forecast_target_date": pd.Timestamp(forecasts["target_date"].max()).strftime("%Y-%m-%d"),
        "last_scored_date": (pd.Timestamp(scores["Date"].max()).strftime("%Y-%m-%d") if len(scores) else None),
        "scored_days": int(len(scores)),
        "model_version": int(status["model_version"]),
        "model_trained_to": str(status["model_trained_to"]),
    }
    return results

def build_dashboard(forecasts_path, scores_path, actual, benchmark, days=DEFAULT_REVIEW_DAYS):
    """Assemble every section of the dashboard file."""
    forecasts = pd.read_parquet(forecasts_path)
    scores = pd.read_parquet(scores_path)
    updated_at = datetime.now(timezone.utc)

    results ={
        "schema_version": SCHEMA_VERSION,
        "latest_forecast": get_latest_forecast(forecasts_path),
        "daily_scores": get_daily_scores(scores_path),
        "day_review": build_day_review(forecasts, scores, actual, benchmark, days),
        "updated_at": updated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_status": check_pipeline_status(forecasts, scores, updated_at),
    }
    return results


def write_dashboard(dashboard, output_path):
    """
    Write the dashboard dictionary as JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(dashboard, file, allow_nan=False, separators=(",", ":"))
    return output_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write the dashboard JSON file.")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS_PATH)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--days", type=int, default=DEFAULT_REVIEW_DAYS)
    args = parser.parse_args()

    actual = pd.read_parquet(DEMAND_PATH)
    benchmark = build_day_ahead_benchmark()

    dashboard = build_dashboard(
        args.forecasts,
        args.scores,
        actual,
        benchmark,
        args.days
    )

    path = write_dashboard(dashboard, args.output)

    print(
        f"Dashboard saved to {path}: "
        f"{len(dashboard['daily_scores'])} scored days, "
        f"{len(dashboard['day_review'])} half-hourly records."
        )
