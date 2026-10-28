# Predicting Daily Building Energy Use with Machine Learning

IFJC4002 Practical Artificial Intelligence with Python · Faculty of Intelligent Computing, Universiti Malaysia Perlis
Case 4: Building energy consumption prediction

Group: Kitthakarn Sawasdipab, Narawit Midsalae, Arissara Arree, Nattawut Kamchoo, Phakchira Nunnum

Live app: https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/
Repository: https://github.com/kitthakarn-sawasdipab/AI-Project

> **Draft status.** All numbers come from `notebooks/analysis.ipynb`, `docs/figures/results.json` and `models/metadata.json`.
> Items marked **[GROUP]** need the group's own input or evidence (screenshots, manual tests on the deployed app, decisions).
> The group must read, check and rewrite this text in its own words before submitting.
> **Important:** the data are synthetic, so every accuracy figure below describes how well the models recover a simulation, not how they would perform on real buildings.

---

## 1. Problem definition and dataset analysis (CO1)

### 1.1 Problem, user and benefit
A facility manager or energy analyst planning a day of operation often needs an estimate of how much electricity a building will use, for example to budget, to compare with similar buildings, or to size cooling plans. This project builds a regression model that predicts a building's **daily electricity use (kWh/day)** from seven inputs known in advance, and serves it as a dashboard so a non-programmer can enter a building and get an estimate. The case is different from the household electricity example shown in class: it covers commercial and institutional buildings of four types.

| Item | Definition |
|---|---|
| Intended user | Facility manager or energy analyst |
| Target | Daily electricity use of one building on one occupied day, kWh/day |
| Task type | Regression on tabular data |
| Inputs | Floor area (m²), planned occupancy (people), operating hours (h/day), forecast outdoor temperature (°C), building type (Office, Retail, School, Library), cooling efficiency (COP), window-to-wall ratio (0–1) |
| Prediction setting | A building described only by these inputs; no past readings are used |

### 1.2 Dataset
The data are the course file `04_Building_Energy_Consumption.xlsx` (IFJC4002, Case 4). Its Guide sheet states that it is **synthetic teaching data** (generation seed 20261432): 2,000 simulated occupied days, one for each of 2,000 distinct buildings, "not evidence of real-world accuracy". The identifier `record_id` is excluded from the model.

| Property | Value |
|---|---|
| Rows | 2,000 (2,000 unique ids), one per building |
| Missing values | 16 blank `floor_area_m2` and 16 blank `planned_occupancy`, in 32 different rows (1.6% of rows); target and other inputs complete |
| Values outside the documented ranges | none (checked for area, hours, COP and window ratio) |
| Building types | Office 514, Library 511, School 506, Retail 469: well balanced |
| Target | min 128.9, median 3,835, mean 4,396, max 18,676 kWh/day |

### 1.3 Target distribution and unusual values
Daily energy is moderately right-skewed (skewness 1.05; standard deviation 70% of the mean; Figure 1). The 1st and 99th percentiles are 232 and 13,385 kWh/day. Energy per floor area has a median of 0.65 kWh/m²/day (1st–99th percentile 0.33 to 1.58). No values are impossible; the largest buildings simply use the most energy. The 32 blank inputs are the only data quality problem. Their number is too small to say whether the blanks relate to the target (the mean energy of rows with a blank occupancy is lower, 3,151 against 4,407 kWh/day, but this rests on 16 rows).

![Target distribution](figures/fig_target_distribution.png)
*Figure 1. Daily energy use (left), its log10 (middle) and energy per m² (right).*

### 1.4 Features and their relation to the target
Spearman rank correlations with daily energy: floor area **+0.86**, planned occupancy **+0.69**, outdoor temperature +0.29, operating hours +0.27, window-to-wall ratio +0.06, cooling efficiency −0.06 (Figure 2). Four points matter for modelling:
- **Floor area and occupancy are strongly related to each other** (Spearman +0.78), so they carry overlapping information.
- **Spread grows with size.** In the scatter plots the vertical spread widens as area and occupancy increase, a sign that effects are multiplicative rather than additive.
- **Building type shifts energy.** Retail uses the most (mean 5,503 kWh/day, median 0.84 kWh/m²/day) and School the least (3,596; 0.52), although the median areas are similar (about 6,000 m²), so the difference is not caused by size.
- **Weak marginal effects can still matter in combination.** Cooling efficiency and window ratio correlate weakly with energy on their own. The Guide says efficiency affects consumption, so its effect is probably masked by the stronger inputs; we have not tested this for the window ratio.

![Feature relations](figures/fig_feature_relations.png)
*Figure 2. Daily energy against each numeric input, with Spearman correlation.*

![Building type](figures/fig_building_type.png)
*Figure 3. Daily energy and energy per m² by building type.*

### 1.5 Suitable approaches and what the data imply
The target is numeric and the data are tabular, so supervised regression is suitable. Because effects look multiplicative, a **linear model on log-transformed variables** is a natural candidate; a **tree ensemble** is the usual alternative when relations are unknown. Consequences of the data: (1) with only 1,600 training rows, a flexible model may struggle to learn a smooth multiplicative surface (we check this in section 3); (2) the 16 + 16 blanks need imputation fitted on training data only; (3) because the data are simulated, a high score shows that a model recovers the simulation, nothing more.

---

## 2. Solution design and preprocessing (CO2)

### 2.1 Workflow
`Excel file → CSV (unchanged) → random 80/20 split → preprocessing fitted on train only → model → 5-fold cross-validation on train → one test evaluation → save one Pipeline → Streamlit form → validation → prediction → log`.

### 2.2 Features and leakage
All seven inputs are known before the predicted day (the Guide describes occupancy as expected and temperature as a forecast), so none reveals the outcome. `record_id` is excluded. The target is never filled. Preprocessing lives inside the same scikit-learn `Pipeline` as the model, so imputers, scalers and encoders are fitted on training data only; an automated test checks that the imputer statistics equal the training-split medians.

### 2.3 Preprocessing
| Step | Treatment | Reason |
|---|---|---|
| Missing area, occupancy | Median imputation, learned on training data | Only 1.6% of rows; keeps all rows |
| Floor area, occupancy | `log1p`, then standardise | Right-skewed sizes; multiplicative effects |
| Hours, temperature, COP, window ratio | Standardise | Scale for the linear model |
| Building type | One-hot, unknown types ignored | Nominal feature |
| Target (Ridge only) | `log1p` on the target, predictions mapped back with `expm1` | Multiplicative structure, see 3.2 |

### 2.4 Split and validation
Every row is a different building, so a random split cannot place one building in both sets (the Guide also specifies a random split). We used 80% for training (1,600 rows) and 20% for testing (400 rows), seed 42; the mean target is similar in both (4,348 and 4,591 kWh/day) and the test set holds 3 rows with blank area and 3 with blank occupancy. Model selection used 5-fold cross-validation on the training set only. The test set was used only to report results, after the candidate models and their settings were fixed (training was re-run several times for code fixes, with identical results).

---

## 3. Model development and comparison (CO2)

### 3.1 Models
| Model | Settings | Role |
|---|---|---|
| Baseline | Predict the training mean | Reference point |
| Ridge regression | alpha = 1.0, log target, log area and log occupancy | Simple, fast, interpretable |
| Random forest | 300 trees, `min_samples_leaf` = 3, raw target, seed 42 | Non-linear alternative |

### 3.2 How the settings were chosen
Before looking at any test result, we compared seven candidate settings by cross-validation **on the training split only** (`notebooks/cv_explore.py`, output saved in `docs/figures/cv_exploration.txt`):

| Candidate | CV RMSE (kWh, mean ± sd) | CV R² |
|---|---|---|
| Ridge, raw target | 1,439 ± 53 | 0.772 |
| **Ridge, log target** | **593 ± 32** | **0.961** |
| Random forest, raw, leaf 3 | 809 ± 85 | 0.928 |
| Random forest, raw, leaf 10 | 960 ± 93 | 0.898 |
| Random forest, log target, leaf 3 | 832 ± 74 | 0.924 |
| Gradient boosting, raw | 712 ± 58 | 0.944 |
| Gradient boosting, log target | 620 ± 61 | 0.958 |

The target transform mattered more than the algorithm: a log target cut Ridge's error by 59%. That fits a multiplicative relation (energy roughly proportional to area, hours and a cooling term). The random forest was best with a raw target and small leaves, and we kept that setting as the second algorithm. Gradient boosting with a log target came close to Ridge but was not better, so the final comparison keeps the two simpler models.

### 3.3 Results
Test set, 400 buildings (CV on the 1,600 training rows in the last column):

| Model | Test MAE (kWh) | Test RMSE (kWh) | Test R² | Median abs. % error | CV RMSE |
|---|---|---|---|---|---|
| Baseline (mean) | 2,515 | 3,244 | −0.01 | 51.0% | 3,020 ± 124 |
| **Ridge regression (log target)** | **415** | **644** | **0.960** | **7.5%** | **593 ± 32** |
| Random forest | 529 | 871 | 0.928 | 9.5% | 809 ± 85 |

Both models beat the baseline clearly (RMSE lower by 80% for Ridge and 73% for the forest). Ridge is better than the forest on every metric, with RMSE 26% lower and MAE 22% lower, and it is faster: fit time 0.05 s against 0.93 s. On average a Ridge estimate is within about 9% of the true daily use (MAE 9.0% of the mean).

---

## 4. Final model selection, limitations and improvements (CO2)

### 4.1 Selection
The selection rule coded in `src/train.py` is the lowest cross-validation RMSE on the training set. It selects **Ridge regression** (593 against 809), and the gap is many times larger than the fold-to-fold spread (32 and 85 kWh), so the choice does not depend on the test set. The test results agree. Practical constraints point the same way: Ridge is the faster model, trains in a fraction of a second, and its coefficients can be read directly (4.3).

### 4.2 Error analysis
On the 400 test buildings (Ridge):
- **Bias is small:** mean error −29 kWh (very slight under-prediction); 48% of rows are over-predicted.
- **Typical error:** median absolute error 263 kWh, 90th percentile 926 kWh. In percentage terms the median error is −0.4%, the 5th to 95th percentile range is −16% to +23%, **64.5% of predictions are within 10% and 96.5% within 25%**.
- **Relative error is similar everywhere** (MAE as a share of the mean actual use): 8.7–9.2% across the four building types, 7.8–10.6% across floor-area quintiles, 8.3–10.0% across operating-hour bands, 7.6–10.8% across temperature bands and 7.0–11.1% across COP bands (Figure 5). No group is predicted badly. Ridge beat the forest in each of the four building types.
- **Absolute error grows with the level of energy**: MAE is 89 kWh in the lowest predicted-value quintile and 888 kWh in the highest, while the relative error stays near 8–10%. This means the noise is roughly proportional to size, as in a multiplicative model, and explains why a log target helps.
- **Largest errors are mostly Retail buildings with high energy use:** the ten worst predictions are 1,800–3,300 kWh off; six are Retail; eight are under-predictions. The 5% worst predictions carry 24% of the total absolute error.
- **Rows with a blank input:** the test set has 6 such rows, with MAE 787 kWh (19.6% of their mean) against 409 kWh (8.9%) for complete rows. This is consistent with median imputation losing information, but six rows are too few to conclude.

![Predicted vs actual](figures/fig_pred_vs_actual.png)
*Figure 4. Ridge: predicted against actual, residuals against predicted, and percentage error.*

![Error by group](figures/fig_error_by_group.png)
*Figure 5. MAE as a percentage of the mean actual use, by building type, floor-area quintile, operating hours and predicted level.*

### 4.3 What the models use
Permutation importance on the test set (increase in RMSE when a feature is shuffled; Ridge): floor area 2,799 kWh, operating hours 1,193, outdoor temperature 1,043, building type 751, cooling efficiency 313, window ratio 89, **occupancy 30**. The forest ranks them the same way. Occupancy has almost no importance even though it correlates +0.69 with energy, because it is closely related to floor area (+0.78) and adds little once area is known; the same explanation applies to its small Ridge coefficient. This is our interpretation and was not tested, for example by removing area from the inputs.

Ridge coefficients (log scale, for +1 standard deviation of each input): floor area +121% energy, Retail +26%, operating hours +25%, outdoor temperature +23%, window ratio +4%, occupancy +3%, cooling efficiency −7%, School −17%. In plain units, one extra operating hour or one extra °C each raise predicted use by roughly 5%.

![Importance](figures/fig_feature_importance.png)
*Figure 6. Permutation importance, Ridge.*

### 4.4 Limitations
- **The data are simulated.** The models learn the simulation's rules, so the high accuracy (R² 0.96) is not evidence of real-world accuracy. Real buildings add effects absent from the file (equipment, weather history, behaviour, faults).
- **Narrow data coverage.** All temperatures are 23–37 °C, all areas 258–11,989 m², and one day per building. Outside these ranges the model extrapolates by formula. For example at −40 °C it predicts only 111 kWh/day for a median Office, because it scales energy up by about 5% per °C and cannot represent heating; at 1,000,000 m² it gives 570,777 kWh/day. The dashboard flags such inputs as outside the training range, and they should not be trusted.
- Only 1,600 training rows and four building types; no repeated measurements of the same building.
- Blank inputs are filled with the training median, which loses information (4.2).
- Monitoring can show shifts in the inputs but not accuracy; that needs measured outcomes.

### 4.5 Improvements (linked to findings)
1. **Validate on real, measured data.** This is the main gap; the same pipeline can be re-trained on real records.
2. **Tune Ridge's regularisation by cross-validation** and test interactions such as area × operating hours; we kept alpha = 1.0 and did not tune it.
3. **Handle blanks better** (for example a "was missing" indicator or model-based imputation), since rows with a blank input had double the error (4.2), after collecting enough rows to check this.
4. **Report an uncertainty range.** Errors are about ±20% (5th–95th percentile −16% to +23%), so showing a range next to the single value would be more honest.
5. **Check the occupancy data.** If occupancy adds nothing beyond floor area, it may be redundant or measured poorly.

---

## 5. Implementation, dashboard, deployment, testing and monitoring (CO3)

### 5.1 Implementation
Python code in `src/`: `prepare_data.py` (Excel to CSV), `train.py` (training and comparison), `predict.py` (validation and prediction), `monitor.py` (logging). The trained Pipeline (preprocessing plus model) is saved with `joblib` in `models/model.joblib` together with `metadata.json` (feature order, training ranges and medians, categories, results). The dashboard loads the same Pipeline, so the transformations and the feature order are identical to training. Dependencies are in `requirements.txt` and seeds in `src/config.py`.

### 5.2 Dashboard
The Streamlit app has a title, short instructions, a form with six number inputs and a building-type drop-down, a Predict button, and a result labelled in kWh/day. A note states that the model is trained on synthetic data and that estimates are not measured values. **[GROUP: insert screenshots of the Predict tab, an out-of-range warning and the Monitoring tab.]**

**Input validation.** Each numeric input has a physical hard limit: area 1–1,000,000 m², occupancy 0–100,000, operating hours 0–24, temperature −40 to 55 °C, COP 1–10, window ratio 0–1. The building type must be one of the four training categories and no field may be empty. Values inside these limits but outside the range seen in training (for example an area above 11,989 m²) are accepted with a warning that the prediction is less reliable. All inputs sit inside one Streamlit form, and the widgets have no limits of their own, so every value reaches our validation function and a clear message is shown.

### 5.3 Deployment
The app is deployed from the GitHub repository on Streamlit Community Cloud: https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/. **[GROUP: after pushing the new model, reboot the app (Manage app → Reboot app), confirm the new version loads, add the date and the repository access given to the lecturer.]** A reboot is needed because the running process can keep old modules after a push (we saw an `ImportError` from this).

### 5.4 Testing
Full log: `docs/test_log.md`. 35 automated `pytest` tests pass.
- **Dashboard vs notebook:** five test buildings entered through the app form matched the training script's predictions (for example 6,838 vs 6,837.80 kWh/day), differing only by rounding to whole kWh.
- **Boundary:** every numeric input at both of its hard limits is accepted.
- **Invalid:** zero or negative area, negative occupancy, 30 operating hours, temperatures of 99 and −80 °C, COP 0.2, window ratio 2, an unknown building type, text instead of a number and blank values are rejected with a clear message, no prediction is shown and nothing is logged.
- **Missing data and leakage:** the model predicts with blank area and occupancy (median imputation), and its imputer statistics equal the training-split medians.
- **Problem found and fixed:** in the first deployed version, an invalid entry in the browser left the previous result on screen. We moved the inputs into a form without widget limits; local browser tests of the fix passed. **[GROUP: repeat the zero-area, temperature-99 and typical cases on the deployed link and record them in `docs/test_log.md`.]**

### 5.5 Monitoring
Each valid prediction is appended to a log (time, inputs, prediction, list of out-of-range inputs) without personal data. The Monitoring tab shows the number of predictions and the share of requests with inputs outside the training range. **Review trigger:** more than 20% of requests outside the training range, which would suggest that users' buildings differ from the data and the model needs reviewing or retraining. With only a few requests one test input moves this share a lot, so it should be read after a reasonable number of requests. **[GROUP: add observations from the deployed app and keep screenshots, since the free hosting does not keep the log permanently.]** New predictions alone do not show real-world accuracy; that needs measured energy use to compare against.

---

## 6. Conclusion
We built and deployed a regression system that predicts a building's daily electricity use from seven inputs, using the course's synthetic dataset of 2,000 buildings. A log-target Ridge regression, chosen by cross-validation on the training data, reached R² 0.960 and a median error of 7.5% on 400 held-out buildings, clearly ahead of a random forest (R² 0.928) and of the mean baseline. Errors are proportional to building size and similar across building types, and the model relies mainly on floor area, operating hours, temperature and building type. Because the data are simulated, the accuracy shows that the model recovers the simulation, not that it would work on real buildings. The most important next step is to validate it on measured data.

## References
- IFJC4002 course file `04_Building_Energy_Consumption.xlsx` (synthetic teaching data; generation seed 20261432), Faculty of Intelligent Computing, Universiti Malaysia Perlis. Course coordinator: Ir. Dr. Marni Azira Markom.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Streamlit documentation, https://docs.streamlit.io

## Appendices
- A. Contribution record: `docs/contribution_record.md`
- B. Prompt record (at least five prompts): `docs/prompt_record.md` **[GROUP]**
- C. Full test log: `docs/test_log.md`
- D. Analysis notebook: `notebooks/analysis.ipynb`; figures and numbers: `docs/figures/`; candidate comparison: `docs/figures/cv_exploration.txt`
