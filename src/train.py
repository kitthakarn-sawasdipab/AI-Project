"""Train and compare models, then save the best pipeline with its metadata.

Usage:  python -m src.train [--data PATH] [--out-dir DIR]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = set(config.FEATURES + [config.TARGET, config.ID_COL])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=[config.TARGET])  # the target is never filled
    df["building_type"] = df["building_type"].astype(str)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    # Each row is a different building, so a random split keeps buildings apart.
    return train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)


def build_preprocessor() -> ColumnTransformer:
    # Fitted on the training data only, because it lives inside the Pipeline.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    logged = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    plain_numeric = [c for c in config.NUMERIC_FEATURES if c not in config.LOG_FEATURES]
    return ColumnTransformer([
        ("num", numeric, plain_numeric),
        ("num_log", logged, config.LOG_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def log_target(model):
    """Fit on log1p(kWh) and predict back in kWh."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def candidate_models() -> dict:
    return {
        "Baseline (train mean)": DummyRegressor(strategy="mean"),
        # Settings chosen by cross-validation on the training split (notebooks/cv_explore.py).
        "Ridge regression": log_target(Ridge(alpha=1.0)),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=config.SEED),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main(data_path: Path, out_dir: Path) -> None:
    df = load_data(data_path)
    train, test = split(df)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]
    print(f"rows: train={len(train)} test={len(test)}")

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    pipelines, cv_rows = {}, []
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        cv_rows.append({"model": name, "cv_RMSE_mean": -scores.mean(), "cv_RMSE_std": scores.std()})
        pipelines[name] = pipe.fit(X_train, y_train)
    cv_table = pd.DataFrame(cv_rows).set_index("model")

    # Selection uses cross-validation on the training set only; the test set is
    # scored once per model afterwards for reporting.
    best_name = cv_table.drop(index="Baseline (train mean)")["cv_RMSE_mean"].idxmin()
    test_table = pd.DataFrame(
        {name: evaluate(y_test, p.predict(X_test)) for name, p in pipelines.items()}
    ).T
    print("\nCross-validation (train only):\n", cv_table.round(3))
    print("\nTest set:\n", test_table.round(3))
    print(f"\nSelected model: {best_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    best = pipelines[best_name]
    joblib.dump(best, out_dir / "model.joblib")

    meta = {
        "selected_model": best_name,
        "target": config.TARGET,
        "unit": config.UNIT,
        "features": config.FEATURES,
        "numeric_ranges": {
            c: [float(X_train[c].min()), float(X_train[c].max())] for c in config.NUMERIC_FEATURES
        },
        "numeric_medians": {c: float(X_train[c].median()) for c in config.NUMERIC_FEATURES},
        "categories": {c: sorted(X_train[c].unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "n_train": len(train),
        "n_test": len(test),
        "cv_results": cv_table.round(4).to_dict(orient="index"),
        "test_results": test_table.round(4).to_dict(orient="index"),
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Reference cases: re-enter these in the dashboard and compare with the notebook output.
    complete = test.dropna(subset=config.FEATURES)  # the dashboard has no way to enter a blank value
    ref = complete.sample(n=min(5, len(complete)), random_state=config.SEED)[config.FEATURES].copy()
    ref["notebook_prediction"] = best.predict(ref[config.FEATURES]).round(2)
    ref.to_csv(out_dir / "reference_cases.csv", index=False)
    print(f"\nSaved model, metadata and reference cases to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DATA_PATH)
    ap.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    args = ap.parse_args()
    main(args.data, args.out_dir)
