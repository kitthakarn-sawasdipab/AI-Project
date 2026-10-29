# Building Daily Energy Use Prediction (IFJC4002, Case 4)

Predicts a commercial or institutional building's electricity use for one occupied day (kWh/day) from
floor area, planned occupancy, operating hours, outdoor temperature, building type, cooling efficiency
and window-to-wall ratio, and serves the model in a Streamlit dashboard.

> The data are the course's **synthetic** teaching file `04_Building_Energy_Consumption.xlsx`
> (2,000 simulated buildings). Accuracy figures show how well the models recover the simulation,
> not performance on real buildings. See `data/README.md`.

## Group
- Kitthakarn Sawasdipab
- Narawit Midsalae
- Arissara Arree
- Nattawut Kamchoo
- Phakchira Nunnum

## Layout
```
app.py                 Streamlit dashboard (Predict form + Monitoring tab)
src/config.py          columns, paths, seed, split settings
src/prepare_data.py    converts the Excel file to data/processed/building_energy.csv
src/train.py           split, baseline + 2 models, CV, test metrics, saves model
src/predict.py         input validation and prediction (shared by app and tests)
src/monitor.py         prediction log and review trigger
notebooks/analysis.ipynb  EDA, model comparison and error analysis (writes docs/figures/)
notebooks/cv_explore.py   cross-validation comparison of candidate settings (training data only)
tests/                 pytest checks (pipeline, validation, and the app form)
data/                  see data/README.md
docs/                  report (report_final.md, built into report.pdf by build_report.py), outline, test log,
                       contribution and prompt records, figures; screenshots/ is made by capture_screenshots.py
assets/                university logos shown in the dashboard header
models/                created by training: model.joblib, metadata.json, reference_cases.csv
```

## Reproduce
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m src.train          # prints CV and test metrics, saves models/
pytest                       # 35 tests, about 30 s
streamlit run app.py
```
`data/processed/building_energy.csv` is in the repository. To rebuild it from the Excel file see `data/README.md`.
To re-run the analysis notebook:
```bash
pip install -r requirements-dev.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```
Random seed is `config.SEED` (42). Model and preprocessing are saved together as one scikit-learn
Pipeline, so the dashboard applies the same transformations and feature order as training.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository, including `models/` (commit after training).
2. On share.streamlit.io choose the repository, branch `main`, main file `app.py`.
3. Live app: https://building-energy-predict.streamlit.app/
4. After pushing changes to files in `src/`, use **Manage app → Reboot app**, otherwise the running process can keep old modules.
5. Give the lecturer access to the repository if it is private.

## Checking dashboard against notebook
`models/reference_cases.csv` lists complete test rows with the training script's predictions. Enter them
in the dashboard and record the comparison in `docs/test_log.md`.
