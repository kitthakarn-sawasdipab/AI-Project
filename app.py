import streamlit as st

from src import config, monitor
from src.predict import HARD_LIMITS, LABELS, load_artifacts, predict_one, validate_inputs

st.set_page_config(page_title="Building Energy Predictor", page_icon="⚡")

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

st.title("⚡ Building Daily Energy Use Predictor")
st.write(
    "Estimate a commercial or institutional building's electricity use for one occupied day. "
    f"Enter the building details, then press **Predict**. The result is in **{meta['unit']}**."
)
st.caption(
    f"Model: {meta['selected_model']} · trained on {meta['n_train']} simulated building scenarios "
    "(synthetic teaching data). Estimates are not measured values."
)

predict_tab, monitor_tab = st.tabs(["Predict", "Monitoring"])

with predict_tab:
    ranges = meta["numeric_ranges"]
    inputs = {}
    # A form sends all values together when Predict is pressed, so a typed but unconfirmed
    # value is never replaced by an older one. The widgets have no min/max on purpose:
    # every value reaches validate_inputs, which gives a clear error message.
    with st.form("predict_form"):
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
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
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
                    f"prediction is less reliable: {', '.join(LABELS[c] for c in out_of_range)}."
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
