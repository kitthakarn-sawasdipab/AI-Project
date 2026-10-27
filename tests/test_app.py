"""Drive the Streamlit app itself: valid, out-of-range and invalid inputs through the form."""
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.ROOT / "app.py")


@pytest.fixture
def app(artifacts, tmp_path, monkeypatch):
    _, _, model_dir = artifacts
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.csv")
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60).run()


def submit(at, area=None, temp=None):
    if area is not None:
        at.number_input[0].set_value(area)
    if temp is not None:
        at.number_input[1].set_value(temp)
    return at.button[0].click().run()


def test_valid_input_shows_prediction(app):
    at = submit(app, area=3000.0, temp=20.0)
    assert not at.exception and not at.error
    assert "kWh/day" in at.metric[0].value


@pytest.mark.parametrize("area,temp", [(0.0, 20.0), (-5.0, 20.0), (3000.0, 99.0), (3000.0, -80.0)])
def test_invalid_input_shows_error_and_no_prediction(app, area, temp):
    at = submit(app, area=area, temp=temp)
    assert at.error and "must be a number between" in at.error[0].value
    assert not at.metric or "kWh/day" not in at.metric[0].value


def test_out_of_range_input_warns(app):
    at = submit(app, area=900_000.0)
    assert not at.error and at.warning
    assert "kWh/day" in at.metric[0].value


def test_invalid_input_is_not_logged(app):
    submit(app, area=0.0)
    assert not config.LOG_PATH.exists()
