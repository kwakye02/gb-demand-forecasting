from pathlib import Path

import pandas as pd
import pytest

from gb_demand_forecasting.data import download


def test_raw_data_directory_points_to_repository_data_folder():
    project_root = Path(__file__).resolve().parents[1]

    assert download.RAW_DATA_DIRECTORY == project_root / "data" / "raw" / "neso"


def test_validate_dataset_rejects_missing_required_columns(tmp_path):
    file_path = tmp_path / "bad_demand.csv"
    pd.DataFrame(
        {
            "SETTLEMENT_DATE": ["2025-01-01"],
            "SETTLEMENT_PERIOD": [1],
        }
    ).to_csv(file_path, index=False)

    with pytest.raises(RuntimeError, match="Missing columns"):
        download.validate_dataset(file_path)


def test_main_downloads_requested_year(monkeypatch, tmp_path):
    requested_years = []
    downloaded_resources = []
    csv_path = tmp_path / "demanddata_2025.csv"
    pd.DataFrame(
        {
            "SETTLEMENT_DATE": ["2025-01-01"],
            "SETTLEMENT_PERIOD": [1],
            "ND": [31000],
        }
    ).to_csv(csv_path, index=False)

    def fake_get_demand_resources(start_year, end_year):
        requested_years.append((start_year, end_year))
        return [
            {
                "year": start_year,
                "id": "resource-id",
                "url": "https://example.com/demand.csv",
            }
        ]

    def fake_download_resource(resource, output_directory):
        downloaded_resources.append((resource, output_directory))
        return csv_path

    monkeypatch.setattr(download, "get_demand_resources", fake_get_demand_resources)
    monkeypatch.setattr(download, "download_resource", fake_download_resource)

    result = download.main(year=2025, output_directory=tmp_path)

    assert result == [csv_path]
    assert requested_years == [(2025, 2025)]
    assert downloaded_resources == [
        (
            {
                "year": 2025,
                "id": "resource-id",
                "url": "https://example.com/demand.csv",
            },
            tmp_path,
        )
    ]


def test_main_downloads_requested_year_range(monkeypatch, tmp_path):
    requested_years = []
    downloaded_resources = []
    csv_paths = {}

    for year in [2024, 2025]:
        csv_path = tmp_path / f"demanddata_{year}.csv"
        pd.DataFrame(
            {
                "SETTLEMENT_DATE": [f"{year}-01-01"],
                "SETTLEMENT_PERIOD": [1],
                "ND": [31000],
            }
        ).to_csv(csv_path, index=False)
        csv_paths[year] = csv_path

    def fake_get_demand_resources(start_year, end_year):
        requested_years.append((start_year, end_year))
        return [
            {
                "year": year,
                "id": f"resource-id-{year}",
                "url": f"https://example.com/demanddata_{year}.csv",
            }
            for year in range(start_year, end_year + 1)
        ]

    def fake_download_resource(resource, output_directory):
        downloaded_resources.append((resource, output_directory))
        return csv_paths[resource["year"]]

    monkeypatch.setattr(download, "get_demand_resources", fake_get_demand_resources)
    monkeypatch.setattr(download, "download_resource", fake_download_resource)

    result = download.main(start_year=2024, end_year=2025, output_directory=tmp_path)

    assert result == [csv_paths[2024], csv_paths[2025]]
    assert requested_years == [(2024, 2025)]
    assert downloaded_resources == [
        (
            {
                "year": 2024,
                "id": "resource-id-2024",
                "url": "https://example.com/demanddata_2024.csv",
            },
            tmp_path,
        ),
        (
            {
                "year": 2025,
                "id": "resource-id-2025",
                "url": "https://example.com/demanddata_2025.csv",
            },
            tmp_path,
        ),
    ]


def test_main_rejects_invalid_year_range():
    with pytest.raises(ValueError, match="start_year cannot be greater"):
        download.main(start_year=2025, end_year=2024)
