# Report outline (8–12 pages excluding cover, references, appendices)

Marks in brackets are the report rubric weights (40 total).

## 1. Problem definition and dataset analysis (CO1) [Problem 5, Data analysis 5]
- Intended user (e.g. facility manager), practical problem, benefit of predicting daily energy use
- Target (energy_kwh_day, kWh), inputs, task type (regression)
- Dataset: source, licence, filters, size, feature types, missing values, outliers, distributions
- Why regression models suit this case; what the data characteristics imply for performance

## 2. Solution design and preprocessing (CO2) [Design 5]
- Workflow diagram: data → clean → split → preprocess → model → save → app
- Feature choices; excluded fields and leakage risks (same-day meter readings, IDs)
- Missing values, one-hot encoding, scaling; all fitted on training data only
- Split strategy and why (building-wise or chronological)

## 3. Model development and comparison (CO2) [Development 10, Evaluation 10]
- Baseline (train mean), Ridge regression, Random forest; settings and reasons
- Cross-validation on training data for selection; test set used for final reporting
- Table: MAE, RMSE, R² for every model
- Error analysis (by building type, by size, worst cases)

## 4. Final model, limitations, improvements (CO2) [Constraints 5]
- Selection justified by CV results and practical needs (speed, interpretability)
- Limitations; specific feasible improvements linked to the findings

## 5. Implementation, dashboard, deployment, testing, monitoring (CO3)
- Dashboard screenshots, deployment link, test log summary
- Notebook-vs-dashboard agreement on reference cases
- Monitoring record and one review trigger

## 6. Conclusion and references

## Appendices
A. Contribution record (`contribution_record.md`) · B. Prompt record (`prompt_record.md`) · C. Full test log (`test_log.md`)
