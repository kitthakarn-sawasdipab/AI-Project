# Test log

Run against: **local app** (automated, Streamlit `AppTest`, real trained model) · Date: 2026-10-27
Tester: Claude (automated). **The group must repeat sections A–B by hand in the browser, and again on the deployed app, then record the tester names and dates.**

Model under test: random forest (log target), trained on `data/processed/daily_energy.csv`, split by building.

## A. Dashboard vs notebook (from `models/reference_cases.csv`)
| # | Inputs (area m², temp °C, type, day) | Notebook prediction | Dashboard prediction | Match? |
|---|---|---|---|---|
| 1 | 5292.6, 16.6, Education, Friday | 2112.38 | 2,112 kWh/day | Yes |
| 2 | 758.8, 26.8, Public services, Sunday | 251.77 | 252 kWh/day | Yes |
| 3 | 7231.9, 18.9, Entertainment/public assembly, Wednesday | 2240.59 | 2,241 kWh/day | Yes |
| 4 | 1932.8, 16.8, Entertainment/public assembly, Friday | 516.98 | 517 kWh/day | Yes |
| 5 | 2504.0, 24.3, Office, Friday | 588.42 | 588 kWh/day | Yes |

The dashboard rounds to whole kWh, so differences below 0.5 are rounding only.

## B. Typical, boundary and invalid inputs
| # | Type | Input | Expected behaviour | Actual result | Pass? |
|---|---|---|---|---|---|
| 1 | Typical | the five cases in section A | Prediction in kWh/day, no warning | As expected, no errors or warnings | Pass |
| 2 | Boundary | area = 1,000,000 m² | Accepted, out-of-range warning | Predicted 28,380 kWh/day with the out-of-range warning for `floor_area_m2` | Pass |
| 3 | Boundary | area = 1 m², temp = −40 °C, area = 1,000,000 m², temp = 55 °C | Accepted (hard limits are inclusive) | Accepted by `validate_inputs` (pytest `test_boundary_values_are_accepted`); not run through the app UI | Pass (unit level) |
| 4 | Invalid | area = 0 | Error: must be between 1 and 1e+06 | Rejected by `validate_inputs` with that message | Pass (unit level) |
| 5 | Invalid | temp = 99 °C | Error: must be between −40 and 55 | Rejected by `validate_inputs` with that message | Pass (unit level) |
| 6 | Invalid | building type not in training data | Error listing allowed types | Rejected with the list of 16 valid types | Pass (unit level) |
| 7 | Invalid | missing area | Error: area is required | Rejected: "floor_area_m2 is required" | Pass (unit level) |
| 8 | Invalid | area = 0 and temp = 99 typed into the form | Browser form refuses the value, or the app shows an error | Original version: fields turned red but Predict silently used old values (section D). After the form fix the app shows a clear error and no prediction (section D-fix) | Pass after fix (deployed link to be re-checked) |

Rows 4–7 test the validation function directly. Row 8 was checked in a browser on the deployed app (section D).

Automated checks: `pytest` (11 tests) covers reload agreement, boundary and invalid inputs, and that no building appears in both train and test.

## D. Deployed app, checked in a browser
App: https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/ · Date: 2026-10-27 · Run by Claude through the built-in browser (no login needed; the app was open to the public, which also confirms access).
**The group should repeat these checks themselves and add their names and screenshots.**

| # | Input | Expected | Actual | Pass? |
|---|---|---|---|---|
| D1 | Reference case 1: 5292.6 m², 16.6 °C, Education, Friday | 2,112 kWh/day (notebook 2112.38) | **2,112 kWh/day**, no warning | Pass |
| D2 | Floor area = 1,000,000 m² (Education, 16.6 °C, Friday) | Prediction plus out-of-range warning | 28,109 kWh/day with the warning "outside the range seen during training: floor_area_m2" | Pass |
| D3 | Floor area = 0 | Refused | Field turns red with a warning icon and "Press Enter to apply"; the value is not accepted and the previous result (2,112) stays | Pass |
| D4 | Temperature = 99 °C | Refused | Same red field and no accepted value | Pass |
| D5 | Monitoring tab | Shows number of predictions and out-of-range share | Showed 5 predictions logged and 20% out-of-range | Pass |

**Observations and weaknesses found (D1–D5 above were run on the version before the fix below)**
1. **Entering a value needed Enter or leaving the field.** A number typed but not confirmed was not sent to the app; the first attempt (value typed, Predict clicked at once) returned 25,557 kWh/day for the default 40,662 m² instead of the typed value.
2. **Invalid text did not block Predict.** After D3/D4 the Predict button still worked and the app used the last accepted values, so a user who typed 0 m² and pressed Predict got a result for the old inputs without any message.
3. The forest cannot extrapolate: 1,000,000 m² returns only 28,109 kWh/day, much too low for such a building. The out-of-range warning is therefore justified, and the prediction should not be trusted.

### D-fix. Re-test after moving the inputs into a form (local server, real model, browser)
Change: the inputs are now inside `st.form` and the widgets have no min/max, so every value reaches `validate_inputs` when Predict is pressed. Run on a local server on 2026-10-27; **the deployed app shows the fix only after the commit is pushed and Streamlit redeploys. Re-check D-fix on the live link afterwards.**

| # | Input (Predict pressed straight after typing, no Enter) | Expected | Actual | Pass? |
|---|---|---|---|---|
| F1 | 5292.6 m², 16.6 °C, Education, Friday | 2,112 kWh/day | **2,112 kWh/day** (before the fix this gave the old default's result) | Pass |
| F2 | Floor area = 0 | Clear error, no prediction | "Floor area (m²) must be a number between 1 and 1,000,000", no result shown | Pass |
| F3 | Temperature = 99 | Clear error, no prediction | "Outdoor temperature (°C) must be a number between -40 and 55", no old result left on screen | Pass |

Automated: 18 `pytest` tests pass, including `tests/test_app.py`, which drives the app form with valid, out-of-range and invalid inputs (zero, negative area; temperature 99 and −80) and checks that invalid inputs are not logged.

## C. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| 2026-10-27 (local test session) | 6 | 17% (1 of 6) | The one out-of-range request was the 1,000,000 m² boundary test. Below the 20% review trigger. |
| 2026-10-27 (deployed app, after the browser tests in section D) | 5 | 20% (1 of 5) | Again the 1,000,000 m² boundary test. Exactly at, not above, the "more than 20%" trigger. With so few requests one test input moves the share a lot, so the trigger needs a minimum number of requests to be meaningful. |

This record comes from test traffic, not real users. Update it after the deployed app has been used.

Review trigger: more than 20% of logged requests have inputs outside the training range.
Logged predictions do not show real-world accuracy without measured energy use to compare.
