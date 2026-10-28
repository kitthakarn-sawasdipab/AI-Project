"""Scratch: compare candidate settings by cross-validation on the TRAINING split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src import config, train

df = train.load_data(config.DATA_PATH)
tr, _ = train.split(df)  # the test part is never touched here
X, y = tr[config.FEATURES], tr[config.TARGET]
cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
S = config.SEED
options = {
    "Ridge raw": Ridge(alpha=1.0),
    "Ridge log-target": train.log_target(Ridge(alpha=1.0)),
    "RF raw leaf3": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=S),
    "RF raw leaf10": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=S),
    "RF log leaf3": train.log_target(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=S)),
    "GBM raw": GradientBoostingRegressor(random_state=S),
    "GBM log": train.log_target(GradientBoostingRegressor(random_state=S)),
}
for name, m in options.items():
    pipe = Pipeline([("prep", train.build_preprocessor()), ("model", m)])
    rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    print(f"{name:18s} CV RMSE {rmse.mean():8.1f} +- {rmse.std():6.1f} | CV R2 {r2.mean():.3f}")
