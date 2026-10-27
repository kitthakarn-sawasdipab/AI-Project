# Predicting Daily Building Energy Use with Machine Learning

IFJC4002 Practical Artificial Intelligence with Python · Faculty of Intelligent Computing, Universiti Malaysia Perlis
Case 4: Building energy consumption prediction

Group: Kitthakarn Sawasdipab, Narawit Midsalae, Arissara Arree, Nattawut Kamchoo, Phakchira Nunnum

Live app: https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/
Repository: https://github.com/kitthakarn-sawasdipab/AI-Project

> **Draft status.** All numbers come from `notebooks/analysis.ipynb` and `docs/figures/results.json`.
> Items marked **[GROUP]** need the group's own input or evidence (screenshots, manual tests, decisions).
> The group must read, check and rewrite this text in its own words before submitting.

---

## 1. Problem definition and dataset analysis (CO1)

### 1.1 Problem, user and benefit
Facility managers and energy analysts often need a quick estimate of how much electricity a building should use on a given day, for example to compare it with similar buildings, set a benchmark, or notice a meter that reads far from what is expected. This project builds a regression model that predicts a building's **daily electricity use (kWh/day)** from four inputs available without any metering of that building: floor area, outdoor air temperature, building type and day of the week. It is deployed as a dashboard so a non-programmer can enter a building and get an estimate.

The case is different from the household electricity example shown in class: it covers 1,549 non-residential and residential buildings of many types, with floor area and weather data, instead of one household's sub-metering.

| Item | Definition |
|---|---|
| Intended user | Facility manager or energy analyst |
| Target | Daily electricity use of one building, in kWh/day |
| Task type | Regression on tabular data |
| Inputs | Floor area (m²), mean outdoor air temperature (°C), building type (16 classes), day of week |
| Prediction setting | A building the model has never seen (no past readings of that building are used) |

### 1.2 Data source
We use the **Building Data Genome Project 2 (BDG2)**, a public dataset of hourly energy meter readings, building metadata and weather for buildings at multiple sites, released under CC BY-SA 4.0 (Miller et al., 2020). Three files are used: the cleaned electricity meter readings, the building metadata (floor area `sqm`, primary space usage) and hourly weather. Download steps are in `data/README.md`.

Raw data facts (from `results.json`):

| Raw data | Value |
|---|---|
| Buildings in metadata | 1,636 |
| Electricity meters | 1,578 |
| Hourly readings per meter | 17,544 (2016 and 2017) |
| Missing hourly cells | 8.9% |
| Meters with no readings at all | 6 |
| Buildings missing primary space usage | 21 (floor area: none missing) |

### 1.3 Building the daily table
One row is one building on one day. `src/build_dataset.py` does the following:
1. Sums the 24 hourly readings into daily kWh, **only when all 24 readings exist**.
2. Averages hourly air temperature per site per day (needs at least 18 readings).
3. Joins floor area and building type from the metadata.
4. Removes days below 0.005 kWh/m², which we treat as a dead or disconnected meter (2,724 of 92,636 sampled rows, 2.9%, in the table before this filter was added).
5. Samples 60 random days per building (seed 42) to keep training fast.

Result: **92,459 rows, 1,549 buildings, 1 January 2016 to 31 December 2017**, with no missing values in any feature or the target.

*Limits of the data.* Occupancy, operating hours and public holidays are not in the data, so the day of week is the only schedule information. Holidays were dropped because BDG2 has no holiday calendar per site. Weather is per site, not per building.

### 1.4 Target distribution and unusual values
Daily energy is strongly right-skewed (Figure 1): median 1,489 kWh, mean 3,476 kWh, 99th percentile 28,861 kWh and maximum 158,149 kWh. Skewness is 5.10 on the raw scale and −0.53 after a log transform, so a log transform makes the target close to symmetric.

![Target distribution](figures/fig_target_distribution.png)
*Figure 1. Daily energy use (left), its log10 (middle) and energy per m² (right).*

Energy per floor area (kWh/m²/day) has a median of 0.31 with a 1st to 99th percentile range of 0.011 to 3.05, and a maximum of 22.1. Very high values (the top 0.1% of rows exceed 10.6 kWh/m²/day) are probably small buildings with process loads or area errors; we kept them because they are real readings, but they inflate squared-error metrics.

### 1.5 Features and their relation to the target
- **Floor area** is by far the strongest signal: Spearman correlation with daily energy is 0.78 (Figure 3, left).
- **Building type** is unbalanced (Figure 2): Education has 35,915 rows from 603 buildings and Office 17,760, while Utility has 4 buildings and Religious worship 3. Median energy intensity differs widely by type, from 0.033 kWh/m²/day (Parking) to 1.74 (Utility, only 4 buildings).
- **Day of week:** weekday median intensity is 0.32 to 0.33 kWh/m²/day and weekend 0.27, so weekends use about **17% less** (ratio 0.83).
- **Temperature:** the correlation with daily energy is weak (Spearman 0.04; 0.12 with energy per m²). Median intensity is not monotonic: 0.32 below 0 °C, 0.27 between 0 and 20 °C, then 0.37 between 20 and 30 °C. This U-shape suggests heating and cooling loads, but its overall effect across buildings is small.

![Building type](figures/fig_building_type.png)
*Figure 2. Rows per building type (left) and energy per m² by type (right).*

![Feature relations](figures/fig_feature_relations.png)
*Figure 3. Energy vs floor area (log-log), median intensity by temperature band, and by weekday.*

### 1.6 Suitable AI approaches and what the data implies
The target is numeric and the data are tabular, so supervised regression is appropriate. Energy is roughly proportional to floor area (a near-linear relation on log-log axes), which suits a **linear model on log-transformed variables**. Type-specific behaviour and weather effects may interact, which suits a **tree ensemble**. Three consequences of the data shape the design:
- Skewed target and area: fit on log energy and log area, otherwise large buildings dominate the squared error.
- Strong between-building differences but weak within-building signals: a model with only these four features can learn "how big and what type" much better than "what is happening today", which limits accuracy for unseen buildings.
- Unbalanced building types: results for types with a handful of buildings are unreliable.

---

## 2. Solution design and preprocessing (CO2)

### 2.1 Workflow
`daily table → split by building → preprocessing (fitted on train only) → model → cross-validation → test evaluation → save one Pipeline → Streamlit app → logging`.

### 2.2 Features, exclusions and leakage
- **Used:** `floor_area_m2`, `air_temp_c`, `building_type`, `day_of_week`. All are known before the day happens (temperature can come from a forecast).
- **Not used as features:** `building_id` and `date` (used only for splitting and ordering). We did not use any other meter of the same building or any same-day reading, because those could reveal the target.
- **Leakage controls:** preprocessing lives inside the same scikit-learn `Pipeline` as the model, so scalers, imputers and encoders are fitted on training folds only.

### 2.3 Preprocessing
| Step | Treatment | Reason |
|---|---|---|
| Missing values | Median imputer (numeric), most-frequent (categorical) | The daily table has no missing values, but the app and future data could; imputers are learned from training data only |
| Floor area | `log1p`, then standardise | Heavy right tail |
| Temperature | Standardise | Needed for the linear model |
| Building type, day of week | One-hot, unknown categories ignored | Nominal features |
| Target (Ridge and forest) | `log1p` target, predictions mapped back with `expm1` | Skewness 5.1 → −0.5 |

### 2.4 Split and validation
Because the app predicts **buildings it has never seen**, rows were split **by building** (GroupShuffleSplit, 20% test, seed 42): **1,239 buildings / 73,964 rows for training and 310 buildings / 18,495 rows for testing, with zero buildings in both** (checked in the notebook and by an automated test). Median floor area is similar (5,132 vs 5,417 m²). Model selection uses 5-fold GroupKFold cross-validation on the training set; the test set is used once for the final comparison.

A chronological split was not used because the goal is not forecasting the same buildings forward in time.

---

## 3. Model development and comparison (CO2)

### 3.1 Models and settings
| Model | Settings | Role |
|---|---|---|
| Baseline | Predict the training mean | Reference point |
| Ridge regression | alpha = 1.0, log target, log area | Simple, fast, interpretable |
| Random forest | 150 trees, `min_samples_leaf` = 100, `max_features` = 0.7, log target, seed 42 | Non-linear, interactions |

All three use the same split, the same preprocessing and the same evaluation. Seeds are fixed, and the notebook re-runs the training and checks it reproduces the saved results.

**A finding from development.** Our first random forest used `min_samples_leaf` = 5 and the raw target and reached R² ≈ 0 on the test buildings, no better than the baseline. Floor area is nearly unique per building, so small leaves let the forest memorise individual buildings instead of learning general patterns. Larger leaves (100), a log target and log area fixed this. Cross-validation on the training data showed the same problem (CV RMSE 6,054 against 6,135 for the baseline), but the training script also printed test results at each attempt, so the test set was seen while iterating (about three attempts) and is not completely untouched. This should be stated as a limitation of the evaluation. **[GROUP: describe the attempts in your own words.]**

### 3.2 Results
| Model | CV RMSE (train, mean ± sd) | Test MAE (kWh) | Test RMSE (kWh) | Test R² | Median abs. % error | Fit time | Predict time |
|---|---|---|---|---|---|---|---|
| Baseline (mean) | 6,135 ± 1,166 | 3,189 | 5,134 | −0.01 | 150.7% | 0.2 s | 2.7 ms / 1,000 rows |
| Ridge regression | 4,836 ± 929 | **1,683** | **3,524** | **0.52** | 49.8% | 0.3 s | 2.0 ms / 1,000 rows |
| Random forest | **4,800 ± 1,048** | 1,801 | 3,707 | 0.47 | 50.7% | 8.5 s | 10.1 ms / 1,000 rows |

Both models clearly beat the baseline (RMSE down by about 28% to 31%, MAE down by about 44% to 47%). The metrics use kWh/day, the target's own unit. The test-set MAE is 55% (Ridge) and 59% (forest) of the mean daily energy, so a typical estimate is off by about half of the building's use.

![Predicted vs actual](figures/fig_pred_vs_actual.png)
*Figure 4. Random forest: predicted vs actual on test buildings (log-log), error histogram, and log ratio.*

---

## 4. Final model selection, limitations and improvements (CO2)

### 4.1 Selection
The selection rule coded in `src/train.py` is **lowest cross-validation RMSE on the training set**, which picks the **random forest** (4,800 vs 4,836). This is the model saved and deployed. The honest reading is that **the two models are tied**: the CV difference (36 kWh) is far smaller than the fold-to-fold spread (about 1,000 kWh), and on the test buildings Ridge is slightly better on every metric (R² 0.52 vs 0.47). Practical constraints point to Ridge: it is about 27 times faster to train and 5 times faster to predict and its coefficients can be read directly. The forest's advantage is that it can capture interactions such as type-specific temperature effects. **[GROUP: decide whether to keep the forest or switch to Ridge, and justify it. Switching only requires changing the selection in `src/train.py`, retraining, and pushing `models/`. If you switch because of the test results, say so openly in the report, because that is selecting with the test set.]**

### 4.2 Error analysis
Computed on the 18,495 test rows from 310 unseen buildings (random forest):
- **Bias is small**: mean error −284 kWh (slight under-prediction); 52.5% of rows are over-predicted.
- **Typical vs large errors:** median absolute error 698 kWh, 90th percentile 5,036 kWh.
- **Errors concentrate in few buildings:** the worst 5% of test buildings (15 of 310) account for **33% of all absolute error**. The worst cases are large buildings whose usage differs strongly from similar-sized ones, for example a 44,382 m² education building averaging 1,918 kWh/day that the model over-predicts by about 21,900 kWh/day, and a 37,161 m² public-services building averaging 25,740 kWh/day that is under-predicted by about 17,400.
- **By building type (Figure 5):** errors are 54% to 63% of the mean for the large types (Education 2,449 kWh MAE; Office 1,716; Entertainment 1,031; Public services 1,753). Public services is under-predicted by about 1,405 kWh on average. The extreme relative errors for Religious worship (782%) and Technology/science come from a **single test building each**, so they should not be over-interpreted.
- **By size and temperature:** relative error is similar across floor-area quintiles (55% to 65%), temperature bands (47% to 64%) and weekdays (56% to 61%), so there is no single size, season or day where the model fails badly; the problem is between-building variation.

![Error by group](figures/fig_error_by_group.png)
*Figure 5. Random forest MAE as a percentage of the mean actual use, by type, floor-area quintile and temperature band.*

**Practical consequences.** Estimates are useful as a rough benchmark ("is this building in the right order of magnitude for its size and type?") but not for billing, budgeting or detecting a small fault, because a typical error is about 50% of the daily value.

### 4.3 What the model actually uses
Permutation importance on 5,000 test rows (increase in RMSE when a feature is shuffled): **floor area 2,522 kWh, building type 447 kWh, day of week 9 kWh, temperature 4 kWh.** In other words the model works almost only from size and type; weather and weekday add almost nothing for unseen buildings, even though weekdays differ by 17% in the descriptive analysis. A plausible reason is that these effects are small compared with the differences between buildings, which none of the four features explain. This is an interpretation; we have not tested it directly.

![Importance](figures/fig_feature_importance.png)
*Figure 6. Permutation importance.*

### 4.4 Limitations
- Only four features; no occupancy, operating hours, building age, HVAC type or holidays.
- Predicts unseen buildings, so accuracy is limited (R² ≈ 0.5); adding a building's own history would be far more accurate but needs a different use case.
- Data cover 2016 and 2017 and the BDG2 sites; results may not transfer to other climates or building stock (including tropical Malaysia).
- Sampled 60 days per building; rare days (extreme weather) are thinly represented.
- Several building types have only a few buildings, so their results are unreliable.
- Monitoring can show shifts in inputs, but it cannot show real-world accuracy without measured outcomes.

### 4.5 Improvements (linked to findings)
1. **Add building descriptors** from the metadata (for example year built, number of floors, occupants) if they are sufficiently complete, since floor area and type explain most of the signal and between-building variation is the main error source.
2. **Use a building's own recent history** (for example last week's use) for buildings that are already metered; this targets the 33% of error from a few unusual buildings.
3. **Try gradient boosting and tune with grouped cross-validation**, as the forest and Ridge are tied.
4. **Report prediction intervals** (for example quantile regression) so users see the uncertainty implied by the 50% typical error.
5. **Train on more days** (`--days-per-building 0`) and add a holiday calendar per site.

---

## 5. Implementation, dashboard, deployment, testing and monitoring (CO3)

### 5.1 Implementation
Python code in `src/`: `build_dataset.py` (data), `train.py` (training and comparison), `predict.py` (validation and prediction), `monitor.py` (logging). The trained Pipeline (preprocessing plus model) is saved with `joblib` in `models/model.joblib` together with `metadata.json` (feature order, training ranges, categories, results). The dashboard loads the same Pipeline, so the transformations and the feature order are identical to training. Dependencies are in `requirements.txt` and seeds in `src/config.py`.

### 5.2 Dashboard
The Streamlit app has a title, short instructions, number inputs for floor area and temperature, drop-downs for building type and day, a Predict button and a result labelled in kWh/day. **[GROUP: insert screenshots of the Predict tab, an out-of-range warning and the Monitoring tab.]**

**Input validation.** Floor area must be 1 to 1,000,000 m² and temperature −40 to 55 °C; building type must be one of the 16 training categories; empty values are rejected. Values inside the hard limits but outside the range seen in training (floor area 34 to 81,290 m²; temperature −24.8 to 40.5 °C) are accepted with a warning that the prediction is less reliable.

### 5.3 Deployment
The app is deployed from the GitHub repository on Streamlit Community Cloud: https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/. **[GROUP: add the date deployed and the repo access given to the lecturer.]**

### 5.4 Testing
Full log: `docs/test_log.md`.
- **Dashboard vs notebook:** five test cases entered in the dashboard matched the training script's predictions exactly (for example 2,112 vs 2,112.38 kWh/day), differing only by rounding to whole kWh.
- **Boundary:** a floor area of 1,000,000 m² was accepted and returned a prediction with the out-of-range warning.
- **Invalid:** zero floor area, temperature 99 °C, an unknown building type and missing values are rejected by the validation function with a clear message (11 automated `pytest` tests pass).
- **Not yet verified:** that the browser form itself refuses invalid entries on the deployed app. **[GROUP: test this manually and record it in `docs/test_log.md` row 8.]**

### 5.5 Monitoring
Each prediction is appended to a log (timestamp, inputs, prediction, list of out-of-range inputs) with no personal data. The Monitoring tab shows the number of predictions and the share of requests with inputs outside the training range. **Review trigger:** more than 20% of requests outside the training range, which would suggest that users' buildings differ from the data and the model needs reviewing or retraining. In our test session 6 predictions were logged and 1 (17%) was out of range, below the trigger. **[GROUP: add observations from the deployed app; note that the free hosting does not keep the log permanently, so keep screenshots.]** New predictions alone do not show real-world accuracy; that needs measured energy use to compare against.

---

## 6. Conclusion
We built a regression system that predicts daily building electricity use from floor area, temperature, building type and day of week, using BDG2 data for 1,549 buildings. Splitting by building gave a realistic estimate for unseen buildings: Ridge regression and a random forest both clearly beat the mean baseline (R² of 0.52 and 0.47), but the typical error is about half of the daily use because the model sees only size and type. The forest was selected by cross-validation and deployed as a validated, monitored Streamlit application; Ridge is a competitive, simpler alternative. The most promising next steps are richer building descriptors and, for already metered buildings, recent history.

## References
- Miller, C., Kathirgamanathan, A., Picchetti, B., et al. (2020). The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data*, 7, 368. **[GROUP: verify the citation details against the BDG2 repository README.]**
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Streamlit documentation, https://docs.streamlit.io
- BDG2 repository (CC BY-SA 4.0), https://github.com/buds-lab/building-data-genome-project-2

## Appendices
- A. Contribution record: `docs/contribution_record.md`
- B. Prompt record (at least five prompts): `docs/prompt_record.md` **[GROUP]**
- C. Full test log: `docs/test_log.md`
- D. Analysis notebook: `notebooks/analysis.ipynb`; figures and numbers: `docs/figures/`
