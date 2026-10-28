"""Pipeline tests on the project dataset: it trains, reloads and agrees with itself, preprocessing
is fitted on training data only, and input validation handles typical, boundary and invalid cases."""
import numpy as np
import pandas as pd
import pytest

from src import config, train
from src.predict import HARD_LIMITS, predict_one, validate_inputs


def valid_case(meta):
    case = dict(meta["numeric_medians"])
    case["building_type"] = meta["categories"]["building_type"][0]
    return case


def test_reloaded_model_matches_reference_cases(artifacts):
    model, meta, model_dir = artifacts
    ref = pd.read_csv(model_dir / "reference_cases.csv")
    for _, r in ref.iterrows():
        got = predict_one(model, meta, r[meta["features"]].to_dict())
        assert got == pytest.approx(r["notebook_prediction"], abs=0.01)


def test_selected_model_beats_the_baseline_on_test(artifacts):
    _, meta, _ = artifacts
    res = meta["test_results"]
    assert res[meta["selected_model"]]["RMSE"] < res["Baseline (train mean)"]["RMSE"]


def test_split_has_no_shared_records():
    tr, te = train.split(train.load_data(config.DATA_PATH))
    assert len(tr) == 1600 and len(te) == 400
    assert set(tr[config.ID_COL]).isdisjoint(te[config.ID_COL])


def test_imputer_is_fitted_on_training_data_only(artifacts):
    model, _, _ = artifacts
    tr, _ = train.split(train.load_data(config.DATA_PATH))
    plain = [c for c in config.NUMERIC_FEATURES if c not in config.LOG_FEATURES]
    fitted = model.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_
    assert fitted == pytest.approx(tr[plain].median().to_numpy())


def test_model_handles_missing_numeric_values(artifacts):
    model, meta, _ = artifacts
    row = pd.DataFrame([valid_case(meta)])
    row.loc[0, ["floor_area_m2", "planned_occupancy"]] = np.nan
    assert np.isfinite(model.predict(row[meta["features"]])).all()


def test_typical_input_is_valid(artifacts):
    _, meta, _ = artifacts
    assert validate_inputs(valid_case(meta), meta)[0] == []


@pytest.mark.parametrize("name", list(HARD_LIMITS))
def test_boundary_values_are_accepted(artifacts, name):
    _, meta, _ = artifacts
    for value in HARD_LIMITS[name]:
        errors, _ = validate_inputs({**valid_case(meta), name: value}, meta)
        assert errors == [], (name, value)


@pytest.mark.parametrize("field,value", [
    ("floor_area_m2", 0), ("floor_area_m2", -5), ("planned_occupancy", -1),
    ("operating_hours", 25), ("outdoor_temperature_c", 99), ("cooling_efficiency_cop", 0.5),
    ("window_to_wall_ratio", 1.5), ("building_type", "spaceship"),
    ("floor_area_m2", None), ("outdoor_temperature_c", "hot"),
])
def test_invalid_inputs_are_rejected(artifacts, field, value):
    _, meta, _ = artifacts
    errors, _ = validate_inputs({**valid_case(meta), field: value}, meta)
    assert errors


def test_out_of_range_input_warns_but_predicts(artifacts):
    model, meta, _ = artifacts
    case = {**valid_case(meta), "floor_area_m2": 900_000.0}
    errors, out_of_range = validate_inputs(case, meta)
    assert errors == [] and "floor_area_m2" in out_of_range
    assert predict_one(model, meta, case) >= 0
