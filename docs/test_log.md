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
| 8 | Invalid | area = 0 or temp = 99 typed into the form | Browser form refuses the value, or the app shows an error | **Not verified.** `AppTest` cannot submit out-of-limit values, so the run fell back to the default input | Open |

Rows 4–7 test the validation function directly. Row 8 still needs a manual check in the browser.

Automated checks: `pytest` (11 tests) covers reload agreement, boundary and invalid inputs, and that no building appears in both train and test.

## C. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| 2026-10-27 (test session) | 6 | 17% (1 of 6) | The one out-of-range request was the 1,000,000 m² boundary test. Below the 20% review trigger. |

This record comes from test traffic, not real users. Update it after the deployed app has been used.

Review trigger: more than 20% of logged requests have inputs outside the training range.
Logged predictions do not show real-world accuracy without measured energy use to compare.
