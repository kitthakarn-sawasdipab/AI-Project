"""Central settings. Column names follow the lecturer-provided file 04_Building_Energy_Consumption.xlsx."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "building_energy.csv"
MODEL_DIR = Path(os.environ.get("ENERGY_MODEL_DIR", ROOT / "models"))
LOG_PATH = Path(os.environ.get("ENERGY_LOG_PATH", ROOT / "logs" / "predictions.csv"))

SEED = 42

# One row = one simulated occupied day for one distinct building.
TARGET = "daily_energy_kwh"
ID_COL = "record_id"  # identifier, never a feature

NUMERIC_FEATURES = [
    "floor_area_m2",
    "planned_occupancy",
    "operating_hours",
    "outdoor_temperature_c",
    "cooling_efficiency_cop",
    "window_to_wall_ratio",
]
LOG_FEATURES = ["floor_area_m2", "planned_occupancy"]  # right-skewed sizes, log1p-transformed before scaling
CATEGORICAL_FEATURES = ["building_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # fixed feature order

# Every row is a different building, so a random split cannot put one building in both sets.
TEST_SIZE = 0.2
CV_FOLDS = 5

UNIT = "kWh/day"
