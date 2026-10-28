"""Convert the lecturer's Excel file to data/processed/building_energy.csv (no values are changed or filled).

Usage:  python -m src.prepare_data [--xlsx PATH]
"""
import argparse
from pathlib import Path

import pandas as pd

from . import config

DEFAULT_XLSX = config.ROOT / "data" / "raw" / "04_Building_Energy_Consumption.xlsx"


def main(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Data")
    needed = {config.ID_COL, config.TARGET, *config.FEATURES}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Excel file is missing columns: {sorted(missing)}")
    if df[config.ID_COL].duplicated().any():
        raise ValueError("record_id is not unique")
    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"rows={len(df)}  missing per column: {df.isna().sum()[lambda s: s > 0].to_dict()}")
    print(f"Wrote {config.DATA_PATH}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    main(ap.parse_args().xlsx)
