"""Loading the saved model and predicting for one input, shared by the app and the tests."""
import json

import joblib
import pandas as pd

from . import config

# Hard limits: values outside these are rejected as invalid.
HARD_LIMITS = {"floor_area_m2": (1.0, 1_000_000.0), "air_temp_c": (-40.0, 55.0)}
LABELS = {"floor_area_m2": "Floor area (m²)", "air_temp_c": "Outdoor temperature (°C)",
          "building_type": "Building type", "day_of_week": "Day of week"}


def load_artifacts(model_dir=None):
    model_dir = model_dir or config.MODEL_DIR
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    return model, meta


def validate_inputs(inputs: dict, meta: dict) -> tuple[list, list]:
    """Return (errors, out_of_range). Errors block prediction; out_of_range only warns."""
    errors, out_of_range = [], []
    for col in meta["features"]:
        value = inputs.get(col)
        label = LABELS.get(col, col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            errors.append(f"{label} is required")
            continue
        if col in HARD_LIMITS:
            lo, hi = HARD_LIMITS[col]
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                errors.append(f"{label} must be a number between {lo:,.10g} and {hi:,.10g}")
                continue
            tlo, thi = meta["numeric_ranges"][col]
            if not tlo <= value <= thi:
                out_of_range.append(col)
        elif col in meta["categories"] and str(value) not in meta["categories"][col]:
            errors.append(f"{label} must be one of {meta['categories'][col]}")
    return errors, out_of_range


def predict_one(model, meta: dict, inputs: dict) -> float:
    row = pd.DataFrame([{c: inputs[c] for c in meta["features"]}])  # same feature order as training
    for c in config.CATEGORICAL_FEATURES:
        row[c] = row[c].astype(str)
    return max(float(model.predict(row)[0]), 0.0)  # energy use cannot be negative
