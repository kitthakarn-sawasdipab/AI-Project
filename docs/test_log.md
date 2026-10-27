# Test log

Run against: local app / deployed app (circle one) · Date: TODO · Tester: TODO

## A. Dashboard vs notebook (from `models/reference_cases.csv`)
| # | Inputs | Notebook prediction | Dashboard prediction | Match? |
|---|---|---|---|---|
| 1 | TODO | TODO | TODO | |

## B. Typical, boundary and invalid inputs
| # | Type | Input | Expected behaviour | Actual result | Pass? |
|---|---|---|---|---|---|
| 1 | Typical | mid-range area, 25 °C, office, Monday | Prediction in kWh/day | TODO | |
| 2 | Boundary | area = 1 m², temp = −40 °C | Accepted, warning for out-of-range | TODO | |
| 3 | Boundary | area = 1,000,000 m², temp = 55 °C | Accepted, warning for out-of-range | TODO | |
| 4 | Invalid | area = 0 | Rejected (widget minimum) / error message | TODO | |
| 5 | Invalid | temp = 99 °C | Rejected / error message | TODO | |
| 6 | Invalid | empty or non-numeric area | Rejected / error message | TODO | |

Automated equivalents are in `tests/test_pipeline.py` (`pytest`).

## C. Monitoring observations
| Date | Predictions logged | Out-of-range share | Observation / action |
|---|---|---|---|
| TODO | | | |

Review trigger: more than 20% of logged requests have inputs outside the training range.
Logged predictions do not show real-world accuracy without measured energy use to compare.
