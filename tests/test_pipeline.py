"""Smoke tests on synthetic data: pipeline trains, reloads and agrees with itself,
and input validation handles typical, boundary and invalid cases."""
import pandas as pd
import pytest

from src import config, make_demo_data, train
from src.predict import load_artifacts, predict_one, validate_inputs


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    data = tmp / "demo.csv"
    make_demo_data.make(n_buildings=15, days=60).to_csv(data, index=False)
    train.main(data, tmp / "models")
    model, meta = load_artifacts(tmp / "models")
    return model, meta, tmp / "models"


def valid_case(meta):
    return {
        "floor_area_m2": sum(meta["numeric_ranges"]["floor_area_m2"]) / 2,
        "air_temp_c": 25.0,
        "building_type": meta["categories"]["building_type"][0],
        "day_of_week": meta["categories"]["day_of_week"][0],
    }


def test_reloaded_model_matches_reference_cases(artifacts):
    model, meta, model_dir = artifacts
    ref = pd.read_csv(model_dir / "reference_cases.csv")
    for _, r in ref.iterrows():
        got = predict_one(model, meta, r[meta["features"]].to_dict())
        assert got == pytest.approx(r["notebook_prediction"], abs=0.01)


def test_typical_input_is_valid(artifacts):
    _, meta, _ = artifacts
    assert validate_inputs(valid_case(meta), meta) == ([], [])


def test_boundary_values_are_accepted(artifacts):
    _, meta, _ = artifacts
    for area, temp in [(1.0, -40.0), (1_000_000.0, 55.0)]:
        case = {**valid_case(meta), "floor_area_m2": area, "air_temp_c": temp}
        errors, _ = validate_inputs(case, meta)
        assert errors == []


@pytest.mark.parametrize("field,value", [
    ("floor_area_m2", 0), ("floor_area_m2", -5), ("air_temp_c", 99),
    ("building_type", "spaceship"), ("floor_area_m2", None), ("air_temp_c", "hot"),
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


def test_no_building_shared_between_train_and_test():
    df = make_demo_data.make(n_buildings=15, days=30)
    df["date"] = pd.to_datetime(df["date"])
    tr, te = train.split(df)
    assert set(tr[config.GROUP_COL]).isdisjoint(te[config.GROUP_COL])
