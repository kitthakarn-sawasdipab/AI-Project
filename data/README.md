# Data

## Source
**Building Data Genome Project 2 (BDG2)**, Miller et al., 2020, *Scientific Data*.
Repository: https://github.com/buds-lab/building-data-genome-project-2
Licence: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Cite the source and share
derived data under the same licence. Confirm the exact citation from the repository README before the report.

> Lecturer approval of this dataset is still pending. Change the source if it is not approved.

## Download (into `data/raw/`, not committed)
Git LFS files, so use `media.githubusercontent.com`:
```bash
B=https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data
mkdir -p data/raw
curl -L -o data/raw/metadata.csv           $B/metadata/metadata.csv            # ~0.3 MB
curl -L -o data/raw/weather.csv            $B/weather/weather.csv              # ~19 MB
curl -L -o data/raw/electricity_cleaned.csv $B/meters/cleaned/electricity_cleaned.csv  # ~175 MB
python -m src.build_dataset
```

## What `src/build_dataset.py` does
- Uses the **electricity** meter only (cleaned hourly file), aggregated to daily kWh.
- Keeps a day only if all 24 hourly readings exist; daily temperature needs at least 18 hourly readings.
- Joins floor area (`sqm`) and primary space usage from metadata, and daily mean air temperature by site.
- Drops buildings without floor area, and days under 0.005 kWh/m² (treated as a dead meter; about 3% of days).
- **Samples 60 random days per building** (seed 42) to keep the data and training fast; use `--days-per-building 0` for all days.
- Result: about 92,000 rows, 1,549 buildings, 2016–2017.

## Output: `data/processed/daily_energy.csv`
One row per building per day:

| column | meaning |
|---|---|
| `building_id` | used only to split train/test by building, not a feature |
| `date` | used for ordering, not a feature |
| `energy_kwh_day` | **target**: daily electricity use in kWh |
| `floor_area_m2`, `air_temp_c`, `building_type`, `day_of_week` | features |

## Known limits
- Occupancy, operating hours and holidays are not in the data (day of week is the only schedule proxy).
- Hourly weather timestamps are used as given; time zones are not adjusted.
- Buildings come from a limited set of sites, so performance on other regions is unproven.
