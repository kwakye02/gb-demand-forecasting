# Reference data

## `neso_demand_for_elexon_gaps.csv`

Elexon never published demand for 414 half-hour settlement periods on 20 days between 2020 and
2023. This file holds NESO's published National Demand (`ND`) for those periods. The pipeline
(`gb_demand_forecasting/data/demand/load_elexon_data.py`) uses it to fill the gaps and flags the
filled rows with `is_imputed`.

| Column | Meaning |
|---|---|
| `settlement_date` | UK settlement day (`YYYY-MM-DD`) |
| `settlement_period` | Half-hour period of that day (1-48; 46 or 50 on clock-change days) |
| `neso_nd_mw` | NESO's published National Demand for that period, in MW |

The values are used as published. No adjustment is made for the small average difference between
NESO's and Elexon's figures (Elexon is about 36 MW lower on average).
