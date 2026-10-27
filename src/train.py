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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = set(config.FEATURES + [config.TARGET, config.GROUP_COL, config.DATE_COL])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=[config.TARGET])
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df.sort_values(config.DATE_COL).reset_index(drop=True)


def split(df: pd.DataFrame):
    if config.SPLIT_MODE == "time":
        cutoff = int(len(df) * (1 - config.TEST_SIZE))
        return df.iloc[:cutoff], df.iloc[cutoff:]
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.SEED)
    train_idx, test_idx = next(gss.split(df, groups=df[config.GROUP_COL]))
    return df.iloc[train_idx], df.iloc[test_idx]


def build_preprocessor() -> ColumnTransformer:
    # Fitted on the training data only, because it lives inside the Pipeline.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    return {
        "Baseline (train mean)": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=config.SEED
        ),
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
    print(f"rows: train={len(train)} test={len(test)} (split mode: {config.SPLIT_MODE})")

    if config.SPLIT_MODE == "time":
        cv, groups = TimeSeriesSplit(n_splits=config.CV_FOLDS), None
    else:
        cv, groups = GroupKFold(n_splits=config.CV_FOLDS), train[config.GROUP_COL]

    pipelines, cv_rows = {}, []
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, groups=groups, scoring="neg_root_mean_squared_error"
        )
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
        "categories": {c: sorted(X_train[c].unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "split_mode": config.SPLIT_MODE,
        "n_train": len(train),
        "n_test": len(test),
        "cv_results": cv_table.round(4).to_dict(orient="index"),
        "test_results": test_table.round(4).to_dict(orient="index"),
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Reference cases: re-enter these in the dashboard and compare with the notebook output.
    ref = test.sample(n=min(5, len(test)), random_state=config.SEED)[config.FEATURES].copy()
    ref["notebook_prediction"] = best.predict(ref[config.FEATURES]).round(2)
    ref.to_csv(out_dir / "reference_cases.csv", index=False)
    print(f"\nSaved model, metadata and reference cases to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DATA_PATH)
    ap.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    args = ap.parse_args()
    main(args.data, args.out_dir)
