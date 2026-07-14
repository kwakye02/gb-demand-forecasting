"""
Retrieve and inspect historic GB electricity demand metadata from NESO.

This script:
1. Retrieves metadata for the NESO Historic Demand Data dataset.
2. Selects annual CSV resources from 2009 to 2025.
3. Requests five records from 2009 and 2025.
4. Checks whether the required demand columns are available.
5. Compares the schemas of the two years.
"""

from pprint import pprint

import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# NESO endpoint for retrieving information about a complete dataset.
METADATA_URL = (
    "https://api.neso.energy/api/3/action/package_show"
)

# NESO endpoint for retrieving observations from an individual resource.
DATASTORE_URL = (
    "https://api.neso.energy/api/3/action/datastore_search"
)

# Identifier used by NESO for the Historic Demand Data dataset.
DATASET_ID = "historic-demand-data"

# Complete historical years selected for the first project version.
START_YEAR = 2009
END_YEAR = 2025

# Number of records requested during the initial schema inspection.
SAMPLE_LIMIT = 5

# Inspect the first and last years in the selected period.
SAMPLE_YEARS = {START_YEAR, END_YEAR}

# Columns needed for the initial GB electricity-demand model.
REQUIRED_COLUMNS = {
    "SETTLEMENT_DATE",
    "SETTLEMENT_PERIOD",
    "ND",
}


# ---------------------------------------------------------------------
# Retrieve dataset metadata
# ---------------------------------------------------------------------

print("Requesting NESO Historic Demand Data metadata...")

metadata_response = requests.get(
    METADATA_URL,
    params={"id": DATASET_ID},
    timeout=30,
)

# Stop the script if NESO returns an HTTP error.
metadata_response.raise_for_status()

# Convert the JSON response into a Python dictionary.
metadata = metadata_response.json()

# Confirm that NESO successfully processed the metadata request.
if not metadata.get("success", False):
    raise RuntimeError(
        "NESO metadata request was unsuccessful."
    )

# Extract the complete list of dataset resources.
resources = metadata["result"]["resources"]

print("Metadata request successful.")
print("Total resources found:", len(resources))


# ---------------------------------------------------------------------
# Select the required annual CSV resources
# ---------------------------------------------------------------------

# This list will contain resources from 2009 to 2025 only.
selected_resources = []

for resource in resources:

    # Exclude non-CSV resources, such as the FAQ document.
    if resource.get("format") != "CSV":
        continue

    # Only use resources available through the NESO Datastore API.
    if not resource.get("datastore_active", False):
        continue

    resource_name = resource.get("name", "")

    # Resource names follow a pattern such as:
    # "Historic Demand Data 2009".
    year_text = resource_name.split()[-1]

    # Skip the resource if its name does not end with a year.
    if not year_text.isdigit():
        continue

    year = int(year_text)

    # Keep only complete years in the selected project period.
    if START_YEAR <= year <= END_YEAR:
        selected_resources.append(
            {
                "year": year,
                "name": resource_name,
                "id": resource["id"],
                "url": resource["url"],
            }
        )

# Arrange the selected resources chronologically.
selected_resources.sort(
    key=lambda resource: resource["year"]
)


# ---------------------------------------------------------------------
# Validate the resource selection
# ---------------------------------------------------------------------

expected_years = set(
    range(START_YEAR, END_YEAR + 1)
)

selected_years = {
    resource["year"]
    for resource in selected_resources
}

# Identify any years that were not found in the NESO metadata.
missing_years = expected_years - selected_years

if missing_years:
    raise RuntimeError(
        f"Missing NESO resources for years: "
        f"{sorted(missing_years)}"
    )

print("\nSelected annual resources:")

for resource in selected_resources:
    print(
        resource["year"],
        "|",
        resource["id"],
    )

print("\nResource-selection summary")
print("Number selected:", len(selected_resources))
print("First year:", selected_resources[0]["year"])
print("Last year:", selected_resources[-1]["year"])


# ---------------------------------------------------------------------
# Inspect samples from the first and last selected years
# ---------------------------------------------------------------------

# Store the schemas so that the two years can be compared.
sample_schemas = {}

for resource in selected_resources:
    year = resource["year"]

    # Only inspect the first and last selected years.
    if year not in SAMPLE_YEARS:
        continue

    print(
        f"\nInspecting Historic Demand Data {year}"
    )

    datastore_response = requests.get(
        DATASTORE_URL,
        params={
            "resource_id": resource["id"],
            "limit": SAMPLE_LIMIT,
        },
        timeout=30,
    )

    # Stop if the API returns an HTTP error.
    datastore_response.raise_for_status()

    # Convert the response to a Python dictionary.
    sample_data = datastore_response.json()

    # Confirm that NESO successfully processed the request.
    if not sample_data.get("success", False):
        raise RuntimeError(
            f"NESO Datastore request failed for {year}."
        )

    result = sample_data["result"]

    # Extract the resource schema and sample observations.
    fields = result["fields"]
    records = result["records"]
    total_records = result["total"]

    # Extract the column names from the field metadata.
    field_names = [
        field["id"]
        for field in fields
    ]

    # Store the schema for comparison.
    sample_schemas[year] = set(field_names)

    print("Total records:", total_records)
    print("Column names:", field_names)

    # Check whether the fields required for modelling are present.
    missing_columns = (
        REQUIRED_COLUMNS - set(field_names)
    )

    if missing_columns:
        print(
            "Missing required columns:",
            sorted(missing_columns),
        )
    else:
        print(
            "All required modelling columns are available."
        )

    print("\nFirst five records:")

    for record in records:
        pprint(record)


# ---------------------------------------------------------------------
# Compare the schemas of 2009 and 2025
# ---------------------------------------------------------------------

if SAMPLE_YEARS.issubset(sample_schemas):

    columns_only_in_start_year = (
        sample_schemas[START_YEAR]
        - sample_schemas[END_YEAR]
    )

    columns_only_in_end_year = (
        sample_schemas[END_YEAR]
        - sample_schemas[START_YEAR]
    )

    print("\nSchema comparison")

    print(
        f"Columns only in {START_YEAR}:",
        sorted(columns_only_in_start_year),
    )

    print(
        f"Columns only in {END_YEAR}:",
        sorted(columns_only_in_end_year),
    )

    if (
        not columns_only_in_start_year
        and not columns_only_in_end_year
    ):
        print(
            f"The {START_YEAR} and {END_YEAR} "
            "schemas match."
        )
    else:
        print(
            "The schemas differ and will need to be "
            "standardised during data cleaning."
        )
else:
    print(
        "\nSchema comparison could not be completed "
        "because one or more sample years were unavailable."
    )