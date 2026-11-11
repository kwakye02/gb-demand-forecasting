import pandas as pd
import pytest

from gb_demand_forecasting.data.demand import elexon_demand, load_elexon


def test_settlement_grid_has_46_and_50_periods_on_clock_change_days():
    grid = load_elexon.build_settlement_grid("2026-03-28", "2026-03-30")
    counts = grid.groupby("settlement_date").size()

    assert counts[pd.Timestamp("2026-03-28")] == 48
    assert counts[pd.Timestamp("2026-03-29")] == 46
    assert load_elexon.periods_in_day("2026-10-25") == 50


def test_settlement_grid_stops_at_last_published_period_on_final_day():
    grid = load_elexon.build_settlement_grid("2026-06-01", "2026-06-02", last_period_on_last_day=10)

    assert len(grid) == 48 + 10
    assert grid["settlement_period"].max() == 48


def test_fill_known_gaps_fills_only_recorded_periods_and_flags_them():
    grid = load_elexon.build_settlement_grid("2022-01-01", "2022-01-01").head(4)
    elexon = pd.DataFrame({
        "settlement_date": pd.Timestamp("2022-01-01"),
        "settlement_period": [1, 2],
        "demand_mw": [100.0, 110.0],
    })
    gap_fill = pd.DataFrame({
        "settlement_date": pd.Timestamp("2022-01-01"),
        "settlement_period": [3],
        "fill_mw": [120.0],
    })

    filled = load_elexon.fill_known_gaps(grid, elexon, gap_fill)

    assert filled["demand_mw"].tolist()[:3] == [100.0, 110.0, 120.0]
    assert filled["is_imputed"].tolist() == [False, False, True, False]
    assert pd.isna(filled["demand_mw"].iloc[3])


def test_fill_known_gaps_prefers_elexon_value_when_it_exists():
    grid = load_elexon.build_settlement_grid("2022-01-01", "2022-01-01").head(1)
    elexon = pd.DataFrame({
        "settlement_date": [pd.Timestamp("2022-01-01")],
        "settlement_period": [1],
        "demand_mw": [100.0],
    })
    gap_fill = elexon.rename(columns={"demand_mw": "fill_mw"}).assign(fill_mw=999.0)

    filled = load_elexon.fill_known_gaps(grid, elexon, gap_fill)

    assert filled["demand_mw"].iloc[0] == 100.0
    assert not filled["is_imputed"].iloc[0]


def forecast_versions():
    return pd.DataFrame({
        "settlement_date": [pd.Timestamp("2026-06-10").date()] * 4,
        "settlement_period": [1, 1, 1, 2],
        "start_time_utc": pd.to_datetime(["2026-06-09T23:00Z"] * 3 + ["2026-06-09T23:30Z"]),
        "publish_time_utc": pd.to_datetime([
            "2026-06-08T10:00Z",  # two days before: eligible but older
            "2026-06-09T07:30Z",  # 08:30 UK on the day before: newest eligible for 09:00
            "2026-06-09T08:30Z",  # 09:30 UK: after the 09:00 cutoff
            "2026-06-09T09:30Z",  # period 2 only published after the cutoff
        ]),
        "boundary": "N",
        "forecast_mw": [1000.0, 2000.0, 3000.0, 4000.0],
    })


def test_latest_forecast_before_cutoff_ignores_versions_published_after_cutoff():
    selected = elexon_demand.latest_forecast_before_cutoff(forecast_versions(), cutoff_time="09:00")

    assert selected["settlement_period"].tolist() == [1]
    assert selected["forecast_mw"].tolist() == [2000.0]


def test_latest_forecast_before_cutoff_uses_uk_local_time_in_summer():
    # 08:00 UK local on 2026-06-09 is 07:00 UTC, so the 07:30 UTC version is too late.
    selected = elexon_demand.latest_forecast_before_cutoff(forecast_versions(), cutoff_time="08:00")

    assert selected["forecast_mw"].tolist() == [1000.0]


def test_create_monthly_chunks_covers_range_without_overlap():
    chunks = elexon_demand.create_monthly_chunks("2026-01-15", "2026-03-10")

    assert chunks == [
        ("2026-01-15", "2026-01-31"),
        ("2026-02-01", "2026-02-28"),
        ("2026-03-01", "2026-03-10"),
    ]


def test_fetch_day_ahead_forecasts_rejects_reversed_dates():
    with pytest.raises(ValueError, match="start_date"):
        elexon_demand.fetch_day_ahead_forecasts("2026-02-01", "2026-01-01")
