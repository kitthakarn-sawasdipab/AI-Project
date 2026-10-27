import streamlit as st

from src import config, monitor
from src.predict import HARD_LIMITS, load_artifacts, predict_one, validate_inputs

st.set_page_config(page_title="Building Energy Predictor", page_icon="⚡")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python -m src.train` first.")
    st.stop()

st.title("⚡ Building Daily Energy Use Predictor")
st.write(
    "Estimate a building's energy use for one day. Enter the building and weather details, "
    f"then press **Predict**. The result is in **{meta['unit']}**."
)
st.caption(
    f"Model: {meta['selected_model']} · trained on {meta['n_train']} building-days. "
    "Estimates are not measured values."
)

predict_tab, monitor_tab = st.tabs(["Predict", "Monitoring"])

with predict_tab:
    ranges = meta["numeric_ranges"]
    inputs = {}
    col1, col2 = st.columns(2)
    with col1:
        lo, hi = HARD_LIMITS["floor_area_m2"]
        inputs["floor_area_m2"] = st.number_input(
            "Floor area (m²)", min_value=lo, max_value=hi,
            value=float(round(sum(ranges["floor_area_m2"]) / 2)), step=100.0,
            help=f"Training range: {ranges['floor_area_m2'][0]:.0f}–{ranges['floor_area_m2'][1]:.0f} m²",
        )
        inputs["building_type"] = st.selectbox("Building type", meta["categories"]["building_type"])
    with col2:
        lo, hi = HARD_LIMITS["air_temp_c"]
        inputs["air_temp_c"] = st.number_input(
            "Outdoor temperature (°C)", min_value=lo, max_value=hi,
            value=float(round(sum(ranges["air_temp_c"]) / 2, 1)), step=0.5,
            help=f"Training range: {ranges['air_temp_c'][0]:.1f}–{ranges['air_temp_c'][1]:.1f} °C",
        )
        inputs["day_of_week"] = st.selectbox("Day of week", meta["categories"]["day_of_week"])

    if st.button("Predict", type="primary"):
        errors, out_of_range = validate_inputs(inputs, meta)
        if errors:
            for e in errors:
                st.error(e)
        else:
            pred = predict_one(model, meta, inputs)
            st.metric("Predicted energy use", f"{pred:,.0f} {meta['unit']}")
            if out_of_range:
                st.warning(
                    "These inputs are outside the range seen during training, so the "
                    f"prediction is less reliable: {', '.join(out_of_range)}."
                )
            monitor.log_prediction(inputs, pred, out_of_range)

with monitor_tab:
    st.write("Non-sensitive record of inputs and predictions made through this app.")
    log = monitor.read_log()
    summary = monitor.summarise(log)
    st.metric("Predictions logged", summary["n_predictions"])
    st.metric("Share with out-of-range inputs", f"{summary['out_of_range_share']:.0%}")
    if summary["needs_review"]:
        st.warning("Review trigger: over 20% of requests fall outside the training range. "
                   "Check whether the model needs retraining.")
    st.caption("Logged predictions alone do not show real-world accuracy; that needs measured outcomes.")
    if not log.empty:
        st.dataframe(log.tail(50), width="stretch")
