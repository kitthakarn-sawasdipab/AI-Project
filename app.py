import base64
from pathlib import Path

import streamlit as st

from src import config, monitor
from src.predict import HARD_LIMITS, LABELS, load_artifacts, predict_one, validate_inputs

st.set_page_config(page_title="Building Energy Predictor", page_icon="⚡", layout="wide")

ASSETS = Path(__file__).resolve().parent / "assets"


@st.cache_data
def logo_uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


st.markdown(
    f"""
    <style>
    .brand {{display: flex; align-items: center; gap: 22px; margin: 0 0 .8rem 0;}}
    .brand img {{width: auto; margin: 0;}}
    .brand .sep {{width: 1px; height: 64px; background: #CBD5E1;}}
    </style>
    <div class="brand">
      <img src="{logo_uri('walailak_logo.png')}" alt="Walailak University" style="height: 96px;">
      <div class="sep"></div>
      <img src="{logo_uri('unimap_logo.png')}" alt="Universiti Malaysia Perlis" style="height: 74px;">
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg, #0F766E 0%, #2563EB 100%); color: white;
           padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin: 0; padding: 0; font-size: 2rem;}
    .hero p {margin: .4rem 0 0 0; opacity: .93; font-size: 1.02rem;}
    div[data-testid="stForm"] {border-radius: 14px; background: white; border: 1px solid #D5E0E8;}
    div[data-testid="stMetricValue"] {font-size: 2.4rem;}
    </style>
    <div class="hero">
      <h1>⚡ Building Daily Energy Use Predictor</h1>
      <p>Estimate a commercial or institutional building's electricity use for one occupied day.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# step and display format for each numeric input
NUMBER_UI = {
    "floor_area_m2": dict(step=100.0, format="%.1f"),
    "planned_occupancy": dict(step=10.0, format="%.0f"),
    "operating_hours": dict(step=0.5, format="%.1f"),
    "outdoor_temperature_c": dict(step=0.5, format="%.1f"),
    "cooling_efficiency_cop": dict(step=0.1, format="%.2f"),
    "window_to_wall_ratio": dict(step=0.05, format="%.2f"),
}


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python -m src.train` first.")
    st.stop()

with st.sidebar:
    st.header("About this model")
    res = meta["test_results"][meta["selected_model"]]
    st.markdown(
        f"**{meta['selected_model']}**, trained on {meta['n_train']} simulated buildings.\n\n"
        f"On {meta['n_test']} held-out simulated buildings: R² **{res['R2']:.2f}**, "
        f"average error **{res['MAE']:,.0f} {meta['unit']}**."
    )
    st.caption(
        "The training data are synthetic teaching data, so this accuracy shows how well the model "
        "recovers a simulation, not how it would do on real buildings."
    )
    st.markdown("**How to use**\n\n1. Enter the building details.\n2. Press **Predict**.\n3. Check the warning if an input is unusual.")
    st.caption("Estimates are not measured values.")

predict_tab, monitor_tab = st.tabs(["🔮 Predict", "📈 Monitoring"])

with predict_tab:
    ranges = meta["numeric_ranges"]
    inputs = {}
    left, right = st.columns([3, 2], gap="large")

    with left:
        # A form sends all values together when Predict is pressed, so a typed but unconfirmed
        # value is never replaced by an older one. The widgets have no min/max on purpose:
        # every value reaches validate_inputs, which gives a clear error message.
        with st.form("predict_form"):
            st.subheader("Building details")
            cols = st.columns(2)
            for i, name in enumerate(config.NUMERIC_FEATURES):
                lo, hi = HARD_LIMITS[name]
                tlo, thi = ranges[name]
                with cols[i % 2]:
                    inputs[name] = st.number_input(
                        LABELS[name],
                        value=float(meta["numeric_medians"][name]),
                        help=f"Allowed: {lo:,.10g} to {hi:,.10g}. Training range: {tlo:,.10g} to {thi:,.10g}.",
                        **NUMBER_UI[name],
                    )
            inputs["building_type"] = st.selectbox(LABELS["building_type"], meta["categories"]["building_type"])
            submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    with right:
        st.subheader("Result")
        if not submitted:
            st.info("Fill in the building details and press **Predict**. The estimate appears here.")
        else:
            errors, out_of_range = validate_inputs(inputs, meta)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                pred = predict_one(model, meta, inputs)
                with st.container(border=True):
                    st.metric("Predicted energy use", f"{pred:,.0f} {meta['unit']}")
                    intensity = pred / inputs["floor_area_m2"]
                    typical = meta.get("type_median_intensity", {}).get(inputs["building_type"])
                    delta = f"{(intensity / typical - 1) * 100:+.0f}% vs typical {inputs['building_type']}" if typical else None
                    st.metric("Daily energy per m²", f"{intensity:.2f} kWh/m²", delta=delta, delta_color="off")
                if out_of_range:
                    st.warning(
                        "These inputs are outside the range seen during training, so the "
                        f"prediction is less reliable: {', '.join(LABELS[c] for c in out_of_range)}."
                    )
                monitor.log_prediction(inputs, pred, out_of_range)

with monitor_tab:
    st.write("Non-sensitive record of inputs and predictions made through this app.")
    log = monitor.read_log()
    summary = monitor.summarise(log)
    m1, m2 = st.columns(2)
    m1.metric("Predictions logged", summary["n_predictions"])
    m2.metric("Share with out-of-range inputs", f"{summary['out_of_range_share']:.0%}")
    if summary["needs_review"]:
        st.warning("Review trigger: over 20% of requests fall outside the training range. "
                   "Check whether the model needs retraining.")
    st.caption("Logged predictions alone do not show real-world accuracy; that needs measured outcomes.")
    if not log.empty:
        st.dataframe(log.tail(50), width="stretch")
