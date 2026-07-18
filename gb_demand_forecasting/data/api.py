from pathlib import Path

import requests


METADATA_URL = (
    "https://api.neso.energy/api/3/action/package_show"
)
DATASET_ID = "historic-demand-data"


def get_demand_resources(start_year=2009, end_year=2025):
    """Return NESO demand resources within the selected period."""

    response = requests.get(
        METADATA_URL,
        params={"id": DATASET_ID},
        timeout=30,
    )
    response.raise_for_status()

    metadata = response.json()

    if not metadata.get("success"):
        raise RuntimeError("NESO metadata request failed.")

    selected_resources = []

    for resource in metadata["result"]["resources"]:
        name = resource.get("name", "")
        year_text = name.split()[-1]

        if resource.get("format") != "CSV":
            continue

        if not year_text.isdigit():
            continue

        year = int(year_text)

        if start_year <= year <= end_year:
            selected_resources.append(
                {
                    "year": year,
                    "id": resource["id"],
                    "url": resource["url"],
                }
            )

    selected_resources.sort(key=lambda item: item["year"])

    expected_years = set(range(start_year, end_year + 1))
    available_years = {
        item["year"] for item in selected_resources
    }

    if available_years != expected_years:
        missing = expected_years - available_years
        raise RuntimeError(f"Missing years: {sorted(missing)}")

    return selected_resources


def download_resource(resource, output_directory):
    """Download one NESO CSV unless it already exists."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    file_path = (
        output_directory
        / f"demanddata_{resource['year']}.csv"
    )

    if file_path.exists() and file_path.stat().st_size > 0:
        print(f"{file_path.name} already exists.")
        return file_path

    response = requests.get(
        resource["url"],
        timeout=120,
    )
    response.raise_for_status()

    file_path.write_bytes(response.content)

    print(f"Downloaded {file_path.name}")
    return file_path