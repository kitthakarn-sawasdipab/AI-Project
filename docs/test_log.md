# Test log

Model under test: Ridge regression (log target) trained on `data/processed/building_energy.csv` (lecturer-provided synthetic data).
Date of the checks below: 2026-10-28. Run by Claude with automated tools on the **local** app.

> **The deployed app has not been re-tested with this model.** The live link still runs the earlier version until the new
> commit is pushed and the app is rebooted. **The group must repeat sections A and B by hand on the deployed link,
> add tester names and dates, and attach screenshots.**

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

## C. Browser checks of the form (earlier version of the app)
Before the form fix, typing an invalid value (area 0) turned the box red but the Predict button silently reused the last accepted value and showed an old result. We moved the inputs into `st.form` without widget limits so that
every value reaches the validation function. On a local server with the previous model we confirmed in the browser that a zero area and a temperature of 99 now give a clear error and no stale result, and that a value typed without pressing Enter is used.
**[GROUP: repeat on the deployed link with the current model: zero area, temperature 99, and a typical case, and add screenshots.]**

## D. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| [GROUP] | | | Use the deployed app, then read the Monitoring tab and record the numbers. The free hosting does not keep the log for ever, so take a screenshot. |

Review trigger: more than 20% of logged requests have inputs outside the training range. With very few requests one test input moves the share a lot,
so the trigger is only meaningful after a reasonable number of requests. Logged predictions do not show real-world accuracy without measured energy use to compare.
