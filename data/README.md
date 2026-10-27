# Data

Fill this in once the lecturer approves the dataset.

- **Source and citation:** TODO (e.g. Building Data Genome Project 2)
- **License / permitted use:** TODO
- **Download link and steps:** TODO
- **Filters or sampling applied:** TODO (which meter, which buildings, which dates)

## Expected file

`data/processed/daily_energy.csv`, one row per building per day:

| column | meaning |
|---|---|
| `building_id` | used only to split train/test by building, not a feature |
| `date` | used for ordering, not a feature |
| `energy_kwh_day` | **target**: daily energy use in kWh |
| `floor_area_m2`, `air_temp_c`, `building_type`, `day_of_week`, `is_holiday` | features |

If the approved dataset has different columns, edit `src/config.py`. Do not commit raw files
unless the licence allows redistribution.
