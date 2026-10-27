"""Basic monitoring: append non-sensitive inputs and predictions to a CSV log."""
from datetime import datetime, timezone

import pandas as pd

from . import config


def log_prediction(inputs: dict, prediction: float, out_of_range: list) -> None:
    config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **inputs,
        "prediction": round(float(prediction), 2),
        "out_of_range": ";".join(out_of_range),
    }
    pd.DataFrame([row]).to_csv(
        config.LOG_PATH, mode="a", header=not config.LOG_PATH.exists(), index=False
    )


def read_log() -> pd.DataFrame:
    if not config.LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(config.LOG_PATH)


def summarise(log: pd.DataFrame, review_threshold: float = 0.2) -> dict:
    """Review trigger: share of requests with inputs outside the training range."""
    if log.empty:
        return {"n_predictions": 0, "out_of_range_share": 0.0, "needs_review": False}
    share = float(log["out_of_range"].fillna("").ne("").mean())
    return {
        "n_predictions": int(len(log)),
        "out_of_range_share": share,
        "needs_review": share > review_threshold,
    }
