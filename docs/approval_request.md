# Draft: request for case and dataset approval

**To:** Ir. Dr. Marni Azira Markom
**Subject:** IFJC4002 project – approval request for Case 4 (Building energy consumption prediction)

Dear Dr. Marni,

Our group (members and IDs: TODO) would like to take Case 4, building energy consumption
prediction, and asks for approval of the following dataset and scope.

**Proposed dataset:** Building Data Genome Project 2 (BDG2), public hourly meter readings and
metadata for many buildings, with weather data. Link and licence: TODO (to be confirmed before submission).

**Scope:** electricity meter only, aggregated to one row per building per day.
- Target: daily energy use (kWh), regression
- Inputs: floor area, outdoor air temperature, building type, day of week
  (occupancy and operating hours are not in the data, so day of week is used as a proxy)
- Different from the household electricity example shown in class: it covers many non-residential
  buildings with floor area and weather, not one household's sub-metering.

**Evaluation plan:** split by building so no building is in both train and test; baseline plus two
models (Ridge regression and random forest); MAE, RMSE and R².

**Leakage controls:** no same-day readings from other meters, no identifiers as features.

Please let us know whether this dataset and scope are approved, or if you prefer another dataset.

Thank you,
TODO (group name and members)
