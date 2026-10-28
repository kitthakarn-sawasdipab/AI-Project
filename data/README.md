# Data

## Source
**`04_Building_Energy_Consumption.xlsx`**, provided by the course (IFJC4002, Case 4). The Guide sheet of the file states:
- **Synthetic teaching data.** No real buildings or measurements. It is "not evidence of real-world accuracy".
- 2,000 records, one simulated occupied day for each of 2,000 distinct commercial or institutional buildings (not household data).
- Generation seed 20261432. Missing data: 16 blank values in each of two input columns.
- Instructions: exclude `record_id`, split before fitting imputers, encoders, scalers or models, and use the same feature order in the dashboard.

Cite it in the report as lecturer-provided synthetic data, with the file name above.

## Reproduce
Place the Excel file at `data/raw/04_Building_Energy_Consumption.xlsx` (that folder is not committed), then:
```bash
pip install openpyxl
python -m src.prepare_data     # writes data/processed/building_energy.csv, nothing is changed or filled
```
`data/processed/building_energy.csv` is committed, so the project also runs without the Excel file.

## Columns
| column | role | unit / labels |
|---|---|---|
| `record_id` | identifier, **not a feature** | text |
| `floor_area_m2` | input | m², documented range 250–12,000 (16 blank) |
| `planned_occupancy` | input | people (16 blank) |
| `operating_hours` | input | hours/day, 4–18 |
| `outdoor_temperature_c` | input | °C (forecast daily average; the data span 23–37 °C) |
| `building_type` | input | Office, Retail, School, Library |
| `cooling_efficiency_cop` | input | ratio, 2.0–5.0 |
| `window_to_wall_ratio` | input | 0–1, 0.10–0.70 |
| `daily_energy_kwh` | **target** | kWh/day, 128.9–18,675.8 |

## Known limits
- The data are simulated, so every accuracy figure in this project describes how well the models recover the simulation, not how they would do on real buildings.
- All temperatures are 23–37 °C (cooling-dominated climate); there are no cold-weather examples.
- Each building appears once, so there is no time dimension and no repeated measurements.
- An earlier version of this project used the public BDG2 dataset instead; it was replaced by this file.
