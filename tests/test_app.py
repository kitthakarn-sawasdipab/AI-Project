"""Drive the Streamlit app itself: valid, out-of-range and invalid inputs through the form."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.predict import LABELS

APP = str(config.ROOT / "app.py")


@pytest.fixture
def app(artifacts, tmp_path, monkeypatch):
    _, _, model_dir = artifacts
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.csv")
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60).run()


def submit(at, **values):
    fields = {w.label: w for w in at.number_input}
    for name, value in values.items():
        fields[LABELS[name]].set_value(value)
    return at.button[0].click().run()


def test_default_inputs_show_prediction(app):
    at = submit(app)
    assert not at.exception and not at.error
    assert "kWh/day" in at.metric[0].value


def test_typical_building(app):
    at = submit(app, floor_area_m2=5000.0, planned_occupancy=300.0, operating_hours=10.0)
    assert not at.error and "kWh/day" in at.metric[0].value


@pytest.mark.parametrize("name,value", [
    ("floor_area_m2", 0.0), ("floor_area_m2", -5.0), ("planned_occupancy", -10.0),
    ("operating_hours", 30.0), ("outdoor_temperature_c", 99.0), ("outdoor_temperature_c", -80.0),
    ("cooling_efficiency_cop", 0.2), ("window_to_wall_ratio", 2.0),
])
def test_invalid_input_shows_error_and_no_prediction(app, name, value):
    at = submit(app, **{name: value})
    assert at.error and "must be a number between" in at.error[0].value
    assert not at.metric or "kWh/day" not in at.metric[0].value


def test_out_of_range_input_warns(app):
    at = submit(app, floor_area_m2=50_000.0)
    assert not at.error and at.warning
    assert "kWh/day" in at.metric[0].value


def test_invalid_input_is_not_logged(app):
    submit(app, floor_area_m2=0.0)
    assert not config.LOG_PATH.exists()
