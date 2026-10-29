# Test log

Model under test: Ridge regression (log target) trained on `data/processed/building_energy.csv` (lecturer-provided synthetic data).
Date of the checks below: 2026-10-28 (sections A and B) and 2026-10-29 (section C). Run by Claude with automated tools on the **local** app.

> Sections A and B were run on the local app. **Section C was run on the deployed app** (https://building-energy-predict.streamlit.app/)
> after the new model and dashboard design were pushed and the app rebooted. Screenshots are in `docs/screenshots/`.

## A. Dashboard vs notebook (from `models/reference_cases.csv`, complete test rows)
Entered through the app form (Streamlit `AppTest`, real saved model):

| # | Inputs (area m², occupancy, hours, °C, COP, window ratio, type) | Notebook prediction | Dashboard prediction | Match? |
|---|---|---|---|---|
| 1 | 9634.6, 773, 9.5, 30.6, 4.84, 0.31, Retail | 6837.80 | 6,838 kWh/day | Yes |
| 2 | 566.4, 33, 9.0, 34.0, 3.48, 0.30, School | 336.49 | 336 kWh/day | Yes |
| 3 | 11592.7, 545, 7.4, 30.5, 2.87, 0.21, School | 5389.53 | 5,390 kWh/day | Yes |
| 4 | 11406.9, 312, 11.9, 35.4, 3.21, 0.17, Retail | 12676.13 | 12,676 kWh/day | Yes |
| 5 | 4700.7, 95, 6.3, 32.7, 3.84, 0.23, Office | 2566.35 | 2,566 kWh/day | Yes |

The dashboard rounds to whole kWh, so differences below 0.5 are rounding only. (An earlier reference set contained a row with a blank floor area,
which the form cannot take; the training script now picks reference rows with complete data.)

## B. Typical, boundary and invalid inputs
Automated with `pytest` (35 tests: `tests/test_pipeline.py` for the validation function and `tests/test_app.py` for the app form).

| # | Type | Input | Expected behaviour | Result |
|---|---|---|---|---|
| 1 | Typical | default form values (training medians); a 5000 m², 300 people, 10 h building | Prediction in kWh/day, no warning | Pass |
| 2 | Boundary | each numeric input at its lower and upper hard limit (area 1 and 1,000,000; occupancy 0 and 100,000; hours 0 and 24; temperature −40 and 55; COP 1 and 10; window ratio 0 and 1) | Accepted (limits are inclusive) | Pass (validation function) |
| 3 | Out of range | area 50,000 m² (training range 258–11,989) | Prediction plus a warning that it is less reliable | Pass (app form) |
| 4 | Invalid | area 0 or −5 | Clear error, no prediction, nothing logged | Pass (app form) |
| 5 | Invalid | occupancy −10; hours 30; temperature 99 or −80; COP 0.2; window ratio 2 | Clear error, no prediction | Pass (app form) |
| 6 | Invalid | unknown building type; blank area; text "hot" for temperature | Rejected with a message | Pass (validation function) |
| 7 | Missing data | model asked to predict with blank floor area and occupancy (the training data has such rows) | Imputed with training medians, finite prediction | Pass |
| 8 | Leakage | imputer statistics compared with training-split medians | Equal to the training medians, not the full-data medians | Pass |

Examples of the messages: "Floor area (m²) must be a number between 1 and 1,000,000"; "Outdoor temperature (°C) must be a number between -40 and 55".

**Behaviour at extreme inputs** (Office, other inputs at the training medians; checked by calling the model, not through the browser):
the model gives 7,980 kWh/day at 12,000 m² and 570,777 kWh/day at 1,000,000 m², which scales roughly with size. At −40 °C it gives only
111 kWh/day, because the data only cover 23–37 °C and the model scales energy up by about 5% per °C: it cannot represent heating in the cold. These inputs are inside the hard limits, so they are accepted with an
out-of-range warning, and **the prediction must not be trusted**.

## C. Deployed app, checked in a browser (2026-10-29, current dashboard design)
A scripted Chrome session (Playwright) filled the live form and pressed Predict. It entered the five reference buildings of section A, then three further cases.

| # | Input | Expected | Actual | Pass? |
|---|---|---|---|---|
| C1 | Reference 1: Retail, 9634.6 m² … | 6,838 kWh/day (notebook 6837.80) | 6,838 kWh/day | Pass |
| C2 | Reference 2: School, 566.4 m² … | 336 (notebook 336.49) | 336 kWh/day | Pass |
| C3 | Reference 3: School, 11592.7 m² … | 5,390 (notebook 5389.53) | 5,390 kWh/day | Pass |
| C4 | Reference 4: Retail, 11406.9 m² … | 12,676 (notebook 12676.13) | 12,676 kWh/day | Pass |
| C5 | Reference 5: Office, 4700.7 m² … | 2,566 (notebook 2566.35) | 2,566 kWh/day | Pass |
| C6 | Last case with floor area = 50,000 m² (training range 258–11,989) | Prediction plus out-of-range warning | 25,160 kWh/day and "…less reliable: Floor area (m²)." | Pass |
| C7 | Floor area = 0 | Clear error, no prediction | "Floor area (m²) must be a number between 1 and 1,000,000"; no result card | Pass |
| C8 | Temperature = 99 °C | Clear error, no prediction | "Outdoor temperature (°C) must be a number between -40 and 55"; no result card | Pass |
| C9 | Monitoring tab afterwards | Counts valid predictions only | 6 logged, 17% out of range, no review message | Pass |

Screenshots: `01_dashboard_form.png`, `02_result_typical.png`, `03_warning_out_of_range.png`, `04_error_zero_area.png`, `05_error_temperature.png`, `06_monitoring.png`.

**Notes**
1. The log holds 6 entries: the 5 reference buildings and 1 out-of-range request, one of them out of range (17%). The invalid entries (C7, C8) were not logged, as intended. All 6 are our own test requests. (The log started empty because the app had just been rebooted.)
2. An earlier manual check on the same day, before the redesign, gave the same reference result for case 1. Its first attempt returned a different value because the text "Retail" had been typed into the building-type box without choosing the option, so the form still held the default (Library). Anyone testing by hand must pick the option from the list.
3. In the scripted run the entered values do not need Enter: the form sends everything when Predict is pressed.

## D. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| 2026-10-29 (deployed app, after the checks in C) | 6 | 17% (1 of 6) | The one out-of-range request was the deliberate 50,000 m² test. Below the 20% review trigger, so no review message. With so few requests one test input moves the share a lot. |

Review trigger: more than 20% of logged requests have inputs outside the training range. The trigger is only meaningful after a reasonable number of requests.
The free hosting does not keep the log for ever, so the screenshot is the lasting record. Logged predictions do not show real-world accuracy without measured energy use to compare.
