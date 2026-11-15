import pandas as pd
import pytest

from gb_demand_forecasting.predictions import score

DAY = pd.Timestamp("2026-06-01")


def make_forecast(day=DAY):
    return pd.DataFrame({
        "target_date": day,
        "settlement_period": [1, 2, 3],
        "model_version": 1,
        "forecast_mw": [100.0, 200.0, 300.0],
    })


def make_actual(day=DAY, periods=(1, 2, 3)):
    demand = {1: 110.0, 2: 190.0, 3: 330.0}
    return pd.DataFrame({
        "SETTLEMENT_DATE": day.strftime("%Y-%m-%d"),
        "SETTLEMENT_PERIOD": list(periods),
        "ND": [demand[p] for p in periods],
    })


def make_benchmark(day=DAY):
    return pd.DataFrame({
        "settlement_date": day,
        "settlement_period": [1, 2, 3],
        "elexon_forecast": [100, 200, 330],
    })


def test_score_predictions_errors_are_forecast_minus_actual():
    results = score.score_predictions(make_forecast(), make_actual(), make_benchmark(), DAY)

    assert results["Error_mw"].tolist() == [-10.0, 10.0, -30.0]
    assert results["Benchmark_Error_mw"].tolist() == [-10.0, 10.0, 0.0]


def test_summarize_daily_scores_reports_mae_and_bias_for_model_and_benchmark():
    results = score.score_predictions(make_forecast(), make_actual(), make_benchmark(), DAY)

    row = score.summarize_daily_scores(results).iloc[0]

    assert row["Number_of_Periods"] == 3
    assert row["MAE"] == pytest.approx(16.7, abs=0.05)
    assert row["Bias"] == pytest.approx(-10.0)
    assert row["Elexon_MAE"] == pytest.approx(6.7, abs=0.05)
    assert row["Elexon_Bias"] == pytest.approx(0.0)


def test_score_predictions_rejects_a_day_with_missing_actuals():
    actual = make_actual(periods=(1, 2))

    with pytest.raises(ValueError, match="do not match"):
        score.score_predictions(make_forecast(), actual, make_benchmark(), DAY)


def test_score_new_days_scores_each_day_once_and_skips_days_without_actuals(tmp_path):
    scores_path = tmp_path / "scores.parquet"
    next_day = DAY + pd.Timedelta(days=1)
    forecasts = pd.concat([make_forecast(DAY), make_forecast(next_day)], ignore_index=True)
    benchmark = pd.concat([make_benchmark(DAY), make_benchmark(next_day)], ignore_index=True)

    first = score.score_new_days(forecasts, make_actual(DAY), benchmark, scores_path)
    saved_after_first = pd.read_parquet(scores_path)
    second = score.score_new_days(forecasts, make_actual(DAY), benchmark, scores_path)

    assert first["Date"].tolist() == [DAY]
    assert second["Date"].tolist() == [DAY]
    assert saved_after_first.equals(pd.read_parquet(scores_path))
