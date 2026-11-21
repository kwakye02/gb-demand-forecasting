import json

import pandas as pd
import pytest

from gb_demand_forecasting.reports import export_dashboard

DAY = pd.Timestamp("2026-06-01")
NEXT_DAY = DAY + pd.Timedelta(days=1)


def make_forecasts(days=(DAY, NEXT_DAY)):
    rows = []
    for day in days:
        for period, mw in enumerate([100.0, 200.0, 300.0], start=1):
            rows.append(
                {
                    "target_date": day,
                    "settlement_period": period,
                    "forecast_mw": mw,
                    "issued_at_utc": day - pd.Timedelta(hours=15),
                    "model_trained_to": "2026-05-01",
                    "model_version": 3,
                }
            )
    return pd.DataFrame(rows)


def make_scores(days=(DAY,)):
    return pd.DataFrame(
        {
            "Date": list(days),
            "Model_Version": 3,
            "Number_of_Periods": 3,
            "MAE": 16.7,
            "Bias": -10.0,
            "Elexon_MAE": 6.7,
            "Elexon_Bias": 0.0,
        }
    )


def make_actual(day=DAY):
    return pd.DataFrame(
        {
            "SETTLEMENT_DATE": day.strftime("%Y-%m-%d"),
            "SETTLEMENT_PERIOD": [1, 2, 3],
            "ND": [110.0, 190.0, 330.0],
        }
    )


def make_benchmark(day=DAY):
    return pd.DataFrame(
        {
            "settlement_date": day,
            "settlement_period": [1, 2, 3],
            "elexon_forecast": [100, 200, 330],
        }
    )


def write(frame, tmp_path, name):
    path = tmp_path / name
    frame.to_parquet(path)
    return path


def test_latest_forecast_uses_newest_day_and_computes_peak_and_energy(tmp_path):
    path = write(make_forecasts(), tmp_path, "forecasts.parquet")

    latest = export_dashboard.get_latest_forecast(path)

    assert latest["target_date"] == "2026-06-02"
    assert latest["model_version"] == 3
    assert latest["peak_mw"] == 300.0
    assert latest["peak_period"] == 3
    assert latest["energy_gwh"] == pytest.approx(0.3)
    assert [p["settlement_period"] for p in latest["periods"]] == [1, 2, 3]


def test_latest_forecast_keeps_the_newest_row_when_a_day_was_forecast_twice(tmp_path):
    forecasts = make_forecasts(days=(DAY,))
    rerun = forecasts.copy()
    rerun["forecast_mw"] = rerun["forecast_mw"] + 1
    rerun["issued_at_utc"] = rerun["issued_at_utc"] + pd.Timedelta(hours=1)
    path = write(pd.concat([forecasts, rerun], ignore_index=True), tmp_path, "forecasts.parquet")

    latest = export_dashboard.get_latest_forecast(path)

    assert len(latest["periods"]) == 3
    assert [p["forecast_mw"] for p in latest["periods"]] == [101.0, 201.0, 301.0]


def test_daily_scores_rename_columns_for_the_dashboard(tmp_path):
    path = write(make_scores(), tmp_path, "scores.parquet")

    scores = export_dashboard.get_daily_scores(path)

    assert scores == [
        {
            "date": "2026-06-01",
            "model_version": 3,
            "n_periods": 3,
            "mae": 16.7,
            "bias": -10.0,
            "neso_mae": 6.7,
            "neso_bias": 0.0,
        }
    ]


def test_day_review_joins_forecast_actual_and_neso(tmp_path):
    forecasts = make_forecasts(days=(DAY,))

    review = export_dashboard.build_day_review(
        forecasts, make_scores(), make_actual(), make_benchmark()
    )

    assert review[0] == {
        "date": "2026-06-01",
        "settlement_period": 1,
        "forecast_mw": 100.0,
        "actual_mw": 110.0,
        "neso_mw": 100.0,
    }
    assert len(review) == 3


def test_day_review_skips_days_without_complete_actuals():
    scores = make_scores(days=(DAY, NEXT_DAY))

    review = export_dashboard.build_day_review(
        make_forecasts(), scores, make_actual(), pd.concat([make_benchmark(), make_benchmark(NEXT_DAY)])
    )

    assert {row["date"] for row in review} == {"2026-06-01"}


def test_dashboard_is_valid_json_with_all_sections(tmp_path):
    forecasts_path = write(make_forecasts(), tmp_path, "forecasts.parquet")
    scores_path = write(make_scores(), tmp_path, "scores.parquet")

    dashboard = export_dashboard.build_dashboard(
        forecasts_path, scores_path, make_actual(), make_benchmark()
    )
    output = export_dashboard.write_dashboard(dashboard, tmp_path / "out" / "dashboard.json")
    loaded = json.loads(output.read_text())

    assert set(loaded) == {
        "schema_version",
        "latest_forecast",
        "daily_scores",
        "day_review",
        "updated_at",
        "pipeline_status",
    }
    assert loaded["pipeline_status"]["last_scored_date"] == "2026-06-01"
    assert loaded["pipeline_status"]["scored_days"] == 1


def test_write_dashboard_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        export_dashboard.write_dashboard({"value": float("nan")}, tmp_path / "bad.json")
