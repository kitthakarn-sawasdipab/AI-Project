"""Capture the dashboard screenshots used in the report, and check the reference cases through the live form.

Usage:  python docs/capture_screenshots.py [--url URL] [--out DIR]
Default URL is the deployed app; use http://localhost:8501/ for a local run.
Needs: pip install playwright pandas   (uses the installed Chrome, no browser download)
Every valid request is also written to the app's own log, so run it against the live app sparingly.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import LABELS  # noqa: E402

NUM = ["floor_area_m2", "planned_occupancy", "operating_hours", "outdoor_temperature_c",
       "cooling_efficiency_cop", "window_to_wall_ratio"]


def field(pg, label):
    return pg.locator(f'input[aria-label="{label}"]')


def set_value(pg, label, value):
    f = field(pg, label); f.click(); f.fill(str(value)); f.press("Tab")


def fill_case(pg, vals, btype):
    for k in NUM:
        set_value(pg, LABELS[k], vals[k])
    field(pg, LABELS["building_type"]).click()
    pg.get_by_role("option", name=btype, exact=True).click()


def predict(pg):
    pg.get_by_role("button", name="Predict").click()
    pg.wait_for_timeout(3500)


def result_metric(pg):
    """First metric in the Predict tab (the prediction), or None if no result is shown."""
    card = pg.locator('div[role="tabpanel"]').first.locator('div[data-testid="stMetricValue"]')
    return card.first.inner_text() if card.count() else None


def alerts(pg):
    return [t.strip() for t in pg.locator('div[data-testid="stAlert"]').all_inner_texts()]


def main(url: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(ROOT / "models" / "reference_cases.csv")
    results = {"url": url}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 1250}, device_scale_factor=1.5)
        pg.goto(url, wait_until="domcontentloaded")
        pg.wait_for_selector("text=Building Daily Energy Use Predictor", timeout=90000)
        pg.wait_for_selector('input[aria-label="Floor area (m²)"]', timeout=90000)
        pg.wait_for_timeout(3000)
        txt = pg.inner_text("body")
        results["page_has"] = {"logos": pg.locator(".brand img").count(), "developed_by": "developed by" in txt.lower(),
                               "sidebar": "About this model" in txt}
        print(results["page_has"], flush=True)
        pg.screenshot(path=out / "01_dashboard_form.png", full_page=True)

        cases = []
        for i, r in ref.iterrows():
            fill_case(pg, {k: float(r[k]) for k in NUM}, r.building_type)
            predict(pg)
            cases.append({"case": i + 1, "type": r.building_type, "notebook": float(r.notebook_prediction),
                          "dashboard": result_metric(pg), "alerts": alerts(pg)})
            print(cases[-1], flush=True)
            if i == 0:
                pg.screenshot(path=out / "02_result_typical.png", full_page=True)
        results["reference_cases"] = cases

        set_value(pg, LABELS["floor_area_m2"], 50000); predict(pg)
        results["out_of_range"] = {"metric": result_metric(pg), "alerts": alerts(pg)}; print(results["out_of_range"], flush=True)
        pg.screenshot(path=out / "03_warning_out_of_range.png", full_page=True)

        set_value(pg, LABELS["floor_area_m2"], 0); predict(pg)
        results["invalid_area"] = {"metric": result_metric(pg), "alerts": alerts(pg)}; print(results["invalid_area"], flush=True)
        pg.screenshot(path=out / "04_error_zero_area.png", full_page=True)

        set_value(pg, LABELS["floor_area_m2"], 9634.6); set_value(pg, LABELS["outdoor_temperature_c"], 99); predict(pg)
        results["invalid_temperature"] = {"metric": result_metric(pg), "alerts": alerts(pg)}; print(results["invalid_temperature"], flush=True)
        pg.screenshot(path=out / "05_error_temperature.png", full_page=True)

        pg.get_by_role("tab", name=re.compile("Monitoring")).click()
        pg.wait_for_timeout(2500)
        results["monitoring_text"] = pg.inner_text('div[data-testid="stMainBlockContainer"], main')[:700]
        print(results["monitoring_text"].replace("\n", " | "), flush=True)
        pg.screenshot(path=out / "06_monitoring.png", full_page=True)
        b.close()
    (out / "capture_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", sorted(x.name for x in out.glob("*.png")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://building-energy-predict.streamlit.app/~/+/")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    a = ap.parse_args()
    main(a.url, a.out)
