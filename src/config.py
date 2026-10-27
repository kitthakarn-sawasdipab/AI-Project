"""Central settings. Adapt the column lists once the lecturer approves the dataset."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "daily_energy.csv"
MODEL_DIR = Path(os.environ.get("ENERGY_MODEL_DIR", ROOT / "models"))
LOG_PATH = Path(os.environ.get("ENERGY_LOG_PATH", ROOT / "logs" / "predictions.csv"))

SEED = 42

# One row = one building on one day.
TARGET = "energy_kwh_day"
GROUP_COL = "building_id"  # used only for splitting, never as a feature
DATE_COL = "date"          # used only for chronological splitting

NUMERIC_FEATURES = ["floor_area_m2", "air_temp_c"]
LOG_FEATURES = ["floor_area_m2"]  # heavy-tailed numeric features, log1p-transformed before scaling
CATEGORICAL_FEATURES =["building_type", "day_of_week"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # fixed feature order

# "group": no building appears in both train and test (predicts unseen buildings)
# "time":  train on the earliest dates, test on the latest (forecasting)
SPLIT_MODE = "group"
TEST_SIZE = 0.2
CV_FOLDS = 5

UNIT = "kWh/day"
