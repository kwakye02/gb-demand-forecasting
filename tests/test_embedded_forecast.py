import pandas as pd
import pytest

from gb_demand_forecasting.data import build_dataset, embedded_forecast

RESOURCE_2025 = "fc13df13-2dad-4a1c-b9e3-4569efba4955"
RESOURCE_2026 = "31861619-0b86-47ba-bac2-d008a760af54"
RESOURCE_IDS = {2025: RESOURCE_2025, 2026: RESOURCE_2026, "current": "db6c038f-98af-4570-ab60-24d71ebd0ae5"}


def record(date, period, solar, wind, issued):
    return {
        "SETTLEMENT_DATE": f"{date}T00:00:00",
        "SETTLEMENT_PERIOD": period,
        "EMBEDDED_SOLAR_FORECAST": solar,
        "EMBEDDED_WIND_FORECAST": wind,
        "Forecast_Datetime": issued,
    }


class FakeResponse:
    def __init__(self, records):
        self.records = records

    def raise_for_status(self):
        return None

    def json(self):
        return {"success": True, "result": {"records": self.records}}


class FakeSession:
    """Answers each resource's query with its own records, and keeps the SQL it was sent."""

    def __init__(self, records_by_resource):
        self.records_by_resource = records_by_resource
        self.queries = []

    def get(self, url, params, timeout):
        self.queries.append(params["sql"])
        for resource_id, records in self.records_by_resource.items():
            if resource_id in params["sql"]:
                return FakeResponse(records)
        raise AssertionError("query for an unexpected resource")


def test_cutoff_offset_is_time_between_the_cutoff_and_midnight_of_the_target_day():
    assert embedded_forecast.cutoff_offset("08:30", 1) == "15 hours 30 minutes"
    assert embedded_forecast.cutoff_offset("09:00", 1) == "15 hours 0 minutes"
    assert embedded_forecast.cutoff_offset("08:30", 2) == "39 hours 30 minutes"


def test_month_query_covers_whole_month_and_converts_the_uk_cutoff_to_utc():
    sql = embedded_forecast.build_month_query(RESOURCE_2025, "2025-03-01", "2025-03-31")

    assert "'2025-03-01'" in sql and "'2025-04-01'" in sql  # end date is exclusive
    assert "interval '15 hours 30 minutes'" in sql
    assert "date_trunc('day', \"SETTLEMENT_DATE\")" in sql  # off-schedule malformed rows are excluded
    assert "AT TIME ZONE 'Europe/London'" in sql and "AT TIME ZONE 'UTC'" in sql
    assert RESOURCE_2025 in sql


def test_month_query_rejects_resource_ids_and_dates_that_could_alter_the_query():
    with pytest.raises(ValueError, match="Invalid resource id"):
        embedded_forecast.build_month_query('x"; DROP TABLE y; --', "2025-03-01", "2025-03-31")

    with pytest.raises(ValueError):
        embedded_forecast.build_month_query(RESOURCE_2025, "2025-03-01'; --", "2025-03-31")


def test_normalise_records_converts_types_and_handles_empty_responses():
    df = embedded_forecast.normalise_records(
        [record("2025-03-01", "3", "10.5", "200", "2025-02-28T08:12:00")]
    )

    assert df["settlement_date"].iloc[0] == pd.Timestamp("2025-03-01")
    assert df["settlement_period"].iloc[0] == 3
    assert df["emb_solar_fc"].iloc[0] == 10.5
    assert df["emb_fc_issued_utc"].iloc[0] == pd.Timestamp("2025-02-28T08:12:00Z")
    assert embedded_forecast.normalise_records([]).empty


def test_normalise_records_rejects_responses_with_missing_columns():
    with pytest.raises(ValueError, match="Missing NESO response columns"):
        embedded_forecast.normalise_records([{"SETTLEMENT_DATE": "2025-03-01T00:00:00"}])


def test_newest_per_period_keeps_the_most_recent_issue():
    df = embedded_forecast.normalise_records([
        record("2025-03-01", 1, 5, 100, "2025-02-28T06:12:00"),
        record("2025-03-01", 1, 7, 110, "2025-02-28T08:12:00"),
        record("2025-03-01", 2, 9, 120, "2025-02-28T07:12:00"),
    ])

    result = embedded_forecast.newest_per_period(df)

    assert len(result) == 2
    assert result.loc[result.settlement_period == 1, "emb_solar_fc"].iloc[0] == 7


def test_january_also_queries_the_previous_years_archive():
    assert embedded_forecast.resources_to_query(RESOURCE_IDS, 2026, 1) == [RESOURCE_2026, RESOURCE_2025]
    assert embedded_forecast.resources_to_query(RESOURCE_IDS, 2026, 5) == [RESOURCE_2026]


def test_fetch_combines_both_archives_in_january_and_keeps_the_newest_issue():
    session = FakeSession({
        RESOURCE_2026: [record("2026-01-01", 1, 8, 300, "2025-12-31T08:12:00")],
        RESOURCE_2025: [
            record("2026-01-01", 1, 2, 250, "2025-12-31T07:12:00"),  # older: loses to the newer issue
            record("2026-01-01", 2, 4, 260, "2025-12-31T07:12:00"),  # only the 2025 file has it
        ],
    })

    result = embedded_forecast.fetch_embedded_forecast(
        "2026-01-01", "2026-01-31", session=session, resource_ids=RESOURCE_IDS
    )

    assert len(session.queries) == 2
    assert result["settlement_period"].tolist() == [1, 2]
    assert result["emb_solar_fc"].tolist() == [8, 4]


def test_fetch_returns_an_empty_frame_when_nothing_was_issued():
    session = FakeSession({RESOURCE_2026: []})

    result = embedded_forecast.fetch_embedded_forecast(
        "2026-05-01", "2026-05-31", session=session, resource_ids=RESOURCE_IDS
    )

    assert result.empty and list(result.columns) == embedded_forecast.OUTPUT_COLUMNS


def test_fetch_rejects_ranges_spanning_months_or_reversed_dates():
    session = FakeSession({})

    with pytest.raises(ValueError, match="one calendar month"):
        embedded_forecast.fetch_embedded_forecast("2026-01-15", "2026-02-15", session, RESOURCE_IDS)

    with pytest.raises(ValueError, match="start_date"):
        embedded_forecast.fetch_embedded_forecast("2026-01-20", "2026-01-10", session, RESOURCE_IDS)


def test_dataset_build_attaches_the_forecast_and_leaves_uncovered_periods_empty():
    demand = pd.DataFrame({
        "SETTLEMENT_DATE": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "SETTLEMENT_PERIOD": [1, 2],
        "ND": [30000.0, 29000.0],
    })
    embedded = pd.DataFrame({
        "SETTLEMENT_DATE": ["2026-01-01"],
        "SETTLEMENT_PERIOD": [1],
        "emb_solar_fc": [0.0],
        "emb_wind_fc": [500.0],
        "emb_fc_issued_utc": [pd.Timestamp("2025-12-31T08:12:00Z")],
    })

    merged = build_dataset.add_embedded_forecast(demand, embedded)

    assert len(merged) == 2
    assert merged["emb_wind_fc"].iloc[0] == 500.0
    assert pd.isna(merged["emb_wind_fc"].iloc[1])


def test_rows_whose_settlement_date_is_not_midnight_are_rejected():
    bad = record("2026-06-13", 10, 75, 3451, "2026-06-12T11:54:02")
    bad["SETTLEMENT_DATE"] = "2026-06-13T04:30:00"

    with pytest.raises(ValueError, match="not at midnight"):
        embedded_forecast.normalise_records([bad])


def test_a_forecast_issued_after_the_cutoff_is_refused():
    # cutoff for 13 June is 08:30 BST on 12 June = 07:30 UTC
    late = embedded_forecast.normalise_records([record("2026-06-13", 10, 75, 3451, "2026-06-12T11:54:02")])
    ok = embedded_forecast.normalise_records([record("2026-06-13", 10, 0, 3451, "2026-06-12T07:12:09")])

    embedded_forecast.check_issued_by_cutoff(ok)

    with pytest.raises(ValueError, match="issued after the 08:30 cutoff"):
        embedded_forecast.check_issued_by_cutoff(late)


def test_the_cutoff_check_is_right_in_winter_too():
    # 08:30 GMT on 12 January = 08:30 UTC, so 08:12 is fine and 08:45 is late
    fine = embedded_forecast.normalise_records([record("2026-01-13", 1, 0, 100, "2026-01-12T08:12:00")])
    late = embedded_forecast.normalise_records([record("2026-01-13", 1, 0, 100, "2026-01-12T08:45:00")])

    embedded_forecast.check_issued_by_cutoff(fine)

    with pytest.raises(ValueError):
        embedded_forecast.check_issued_by_cutoff(late)
