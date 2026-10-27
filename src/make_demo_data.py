"""Generate SYNTHETIC data to smoke-test the pipeline. Never use it for the report.

Usage:  python -m src.make_demo_data --out PATH
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n_buildings: int = 40, days: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = np.array(["office", "education", "lodging"])
    rows = []
    for b in range(n_buildings):
        area = rng.uniform(500, 20000)
        btype = rng.choice(types)
        for d, date in enumerate(pd.date_range("2024-01-01", periods=days)):
            temp = 25 + 6 * np.sin(2 * np.pi * d / 365) + rng.normal(0, 2)
            weekend = date.dayofweek >= 5
            holiday = rng.random() < 0.03
            kwh = (
                area * 0.05 * (0.6 if (weekend or holiday) else 1.0)
                + area * 0.02 * max(temp - 24, 0)
                + rng.normal(0, area * 0.01)
            )
            rows.append({
                "building_id": f"b{b:03d}", "date": date, "building_type": btype,
                "floor_area_m2": round(area, 1), "air_temp_c": round(temp, 1),
                "day_of_week": date.day_name(), "is_holiday": int(holiday),
                "energy_kwh_day": round(max(kwh, 0), 1),
            })
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.01, random_state=seed).index, "air_temp_c"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(args.out, index=False)
    print(f"Wrote synthetic demo data to {args.out}")
