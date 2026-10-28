import pytest

from src import config, train
from src.predict import load_artifacts


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Train on the project dataset once and share the saved model with all tests (about 10 s)."""
    out = tmp_path_factory.mktemp("run") / "models"
    train.main(config.DATA_PATH, out)
    model, meta = load_artifacts(out)
    return model, meta, out
