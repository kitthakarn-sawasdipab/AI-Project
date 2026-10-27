import pytest

from src import make_demo_data, train
from src.predict import load_artifacts


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Train on synthetic data once and share the saved model with all tests."""
    tmp = tmp_path_factory.mktemp("run")
    data = tmp / "demo.csv"
    make_demo_data.make(n_buildings=15, days=60).to_csv(data, index=False)
    train.main(data, tmp / "models")
    model, meta = load_artifacts(tmp / "models")
    return model, meta, tmp / "models"
