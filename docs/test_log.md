# Test log

Model under test: Ridge regression (log target) trained on `data/processed/building_energy.csv` (lecturer-provided synthetic data).
Date of the checks below: 2026-10-28. Run by Claude with automated tools on the **local** app.

> Sections A and B were run on the local app. **Section C was run on the deployed app** (https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/)
> after the new model was pushed and the app rebooted, through the built-in browser. **The group should repeat the checks
> in C themselves, add tester names and dates, and attach screenshots.**

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

## C. Deployed app, checked in a browser (2026-10-28)
The page loaded the new version: "Model: Ridge regression · trained on 1600 simulated building scenarios (synthetic teaching data)".

| # | Input (entered in the form, then Predict) | Expected | Actual | Pass? |
|---|---|---|---|---|
| C1 | Reference case 1: 9634.6 m², 773 people, 9.5 h, 30.6 °C, COP 4.84, window 0.31, **Retail** | 6,838 kWh/day (notebook 6837.80) | **6,838 kWh/day**, no warning | Pass |
| C2 | Floor area = 0 | Clear error, no prediction | "Floor area (m²) must be a number between 1 and 1,000,000"; no result shown | Pass |
| C3 | Temperature = 99 °C | Clear error, no prediction | "Outdoor temperature (°C) must be a number between -40 and 55"; no result shown | Pass |
| C4 | Floor area = 50,000 m² (training range 258–11,989) | Prediction plus out-of-range warning | 33,524 kWh/day with "…less reliable: Floor area (m²)." | Pass |
| C5 | Monitoring tab afterwards | Counts the valid predictions only | 3 predictions logged, 33% out of range, and the review-trigger message shown | Pass |

**Notes on this run**
1. In C1 the first attempt gave 5,081 kWh/day instead of 6,838. The cause was the test method, not the app: the text "Retail" had been typed into the building-type box but the option had not been chosen, so the form still held the default (Library). After choosing Retail from the list the result matched. Anyone testing by hand should pick the option from the list.
2. The log holds 3 entries for 3 valid Predict presses (the Library result of 5,081, then C1 and C4). The two invalid presses (C2, C3) were not logged, as intended.
3. Because most of these test inputs were deliberately unusual, the 33% share is not a sign about real use; it shows that the review trigger works.
4. Entering values this way does not need Enter: the form sends everything when Predict is pressed.

## D. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| 2026-10-28 (deployed app, after the checks in C) | 3 | 33% (1 of 3) | The one out-of-range request was the deliberate 50,000 m² test. The share is above 20%, so the tab showed the review message. With three requests one test input moves the share a lot. |
| [GROUP] | | | Add observations from normal use, and take a screenshot, because the free hosting does not keep the log for ever. |

Review trigger: more than 20% of logged requests have inputs outside the training range. With very few requests one test input moves the share a lot,
so the trigger is only meaningful after a reasonable number of requests. Logged predictions do not show real-world accuracy without measured energy use to compare.
