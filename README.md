# Building Daily Energy Use Prediction (IFJC4002, Case 4)

Predicts a building's daily energy use (kWh/day) from floor area, outdoor temperature,
building type, day of week and holiday status, and serves the model in a Streamlit dashboard.

> Status: scaffold. Real data, final features and results are pending lecturer approval of the dataset.
> `src/make_demo_data.py` produces **synthetic** data only to check that the code runs.

## Group
| Name | Student ID | Main tasks |
|---|---|---|
| TODO | TODO | TODO |

## Layout
```
app.py                 Streamlit dashboard (Predict + Monitoring tabs)
src/config.py          columns, split mode, paths, seed
src/train.py           split, baseline + 2 models, CV, test metrics, saves model
src/predict.py         input validation and prediction (shared by app and tests)
src/monitor.py         prediction log and review trigger
src/make_demo_data.py  synthetic data for smoke tests
tests/                 pytest checks (reload agreement, boundary and invalid inputs)
data/                  see data/README.md
docs/                   report outline, test log, contribution and prompt records
models/                created by training: model.joblib, metadata.json, reference_cases.csv
```

## Reproduce
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
# place the approved dataset at data/processed/daily_energy.csv (see data/README.md)
python -m src.train          # prints CV and test metrics, saves models/
pytest                       # smoke tests (use synthetic data)
streamlit run app.py
```
Random seed is `config.SEED` (42). Model and preprocessing are saved together as one scikit-learn
Pipeline, so the dashboard applies the same transformations and feature order as training.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository, including `models/` (commit after training).
2. On share.streamlit.io choose the repository, branch `main`, main file `app.py`.
3. Add the app link here: TODO
4. Give the lecturer access to the repository if it is private.

## Checking dashboard against notebook
`models/reference_cases.csv` lists test rows with the training script's predictions. Enter them
in the dashboard and record the comparison in `docs/test_log.md`.
