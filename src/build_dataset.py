"""Build data/processed/daily_energy.csv from the raw BDG2 files in data/raw/.

One row = one building on one day. Target = daily electricity use (kWh).

Usage:  python -m src.build_dataset [--days-per-building 60]
"""
import argparse

import pandas as pd

from . import config

RAW = config.ROOT / "data" / "raw"
MIN_HOURS_ELEC = 24     # keep only complete days of meter readings
MIN_HOURS_TEMP = 18     # minimum hourly temperature readings for a daily mean
MIN_KWH_PER_M2_DAY = 0.005  # below this the meter is treated as faulty


def build(days_per_building: int) -> pd.DataFrame:
    meta = pd.read_csv(RAW / "metadata.csv", usecols=["building_id", "site_id", "primaryspaceusage", "sqm"])
    meta = meta[meta["sqm"] > 0].dropna(subset=["primaryspaceusage"])

    elec = pd.read_csv(RAW / "electricity_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    elec = elec[elec.columns.intersection(meta["building_id"])]
    # Daily total only when all 24 hourly readings exist.
    daily = elec.groupby(elec.index.normalize()).sum(min_count=MIN_HOURS_ELEC)
    daily.index.name = "date"
    daily = daily.stack().rename(config.TARGET).reset_index()
    daily.columns = ["date", "building_id", config.TARGET]
    daily = daily[daily[config.TARGET] > 0]
    n_before = len(daily)

    weather = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"], usecols=["timestamp", "site_id", "airTemperature"])
    weather = weather.dropna(subset=["airTemperature"])
    temp = (
        weather.groupby(["site_id", weather["timestamp"].dt.normalize()])["airTemperature"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"timestamp": "date", "mean": "air_temp_c"})
    )
    temp = temp[temp["count"] >= MIN_HOURS_TEMP].drop(columns="count")

    df = (
        daily.merge(meta, on="building_id")
        .merge(temp, on=["site_id", "date"])
        .rename(columns={"sqm": "floor_area_m2", "primaryspaceusage": "building_type"})
    )
    # Near-zero use per m² means a dead or disconnected meter, not a real building-day.
    df = df[df[config.TARGET] / df["floor_area_m2"] >= MIN_KWH_PER_M2_DAY]
    print(f"daily rows before merge/filter: {n_before}, after: {len(df)}")
    df["day_of_week"] = df["date"].dt.day_name()
    df["air_temp_c"] = df["air_temp_c"].round(1)
    df[config.TARGET] = df[config.TARGET].round(1)

    if days_per_building:
        shuffled = df.sample(frac=1, random_state=config.SEED)
        df = shuffled[shuffled.groupby("building_id").cumcount() < days_per_building]
    cols = ["building_id", "date", config.TARGET] + config.FEATURES
    return df[cols].sort_values(["date", "building_id"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-per-building", type=int, default=60, help="0 keeps every day")
    args = ap.parse_args()
    out = build(args.days_per_building)
    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.DATA_PATH, index=False)
    print(f"rows={len(out)} buildings={out['building_id'].nunique()} "
          f"dates={out['date'].min().date()}..{out['date'].max().date()}")
    print(out.describe(include="all").T[["count", "unique", "mean", "min", "max"]])
    print(f"Wrote {config.DATA_PATH}")
