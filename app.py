from pathlib import Path
import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inference

ROOT = Path(__file__).resolve().parent
SOFT8 = ["#79C1E4", "#E68282", "#B2D362", "#BAE1F3", "#D4EAF8", "#EECDD5", "#F8E6E4", "#D1E4A6"]
INK = "#30343B"
VERSION = "2026.10.06"

st.set_page_config(page_title="Muscatine | Biogas forecast", layout="wide")
st.markdown("""
<style>
.stMainBlockContainer { max-width: 1220px; padding-top: 2rem; padding-bottom: 2rem; }
h1 { letter-spacing: -.6px; font-size: 2rem !important; }
[data-testid="stSidebar"] { border-right: 1px solid #BAE1F3; }
[data-testid="stMetric"] { border: 1px solid #BAE1F3; border-radius: 10px; padding: 14px 18px; min-height: 102px; }
[data-testid="stMetricValue"] { font-size: 1.8rem; color: #30343B; }
[data-testid="stButton"] button[kind="primary"] { color: #30343B !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #BAE1F3 !important; }
.intro { border-left: 4px solid #79C1E4; padding: 4px 0 4px 20px; margin: 0 0 22px; }
.intro h1 { margin: 0; padding: 0 0 8px; }
.intro p { margin: 0; font-size: 15px; color: #30343B; }
.version { font-size: 12px; padding: 6px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fitted models…")
def bundle():
    return inference.load_bundle()


@st.cache_data(show_spinner=False)
def example_history():
    return pd.read_csv(ROOT / "data" / "history_168h.csv", float_precision="round_trip")


@st.cache_data(show_spinner=False)
def example_plan():
    return pd.read_csv(ROOT / "data" / "recorded_feed_oracle_24h.csv", float_precision="round_trip")


@st.cache_data(show_spinner=False)
def example_batch():
    return pd.read_csv(ROOT / "data" / "history_batch_240h.csv", float_precision="round_trip")


def signature(history, plan):
    text = inference.frame_csv(history) + ("" if plan is None else inference.frame_csv(plan))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chart(result):
    rows = pd.DataFrame(result["rows"])
    has_plan = "feed_m3_h" in rows
    active = "feed" if has_plan else "past"
    band = SOFT8[5] if has_plan else SOFT8[4]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows[active + "_upper_m3_h"], mode="lines", line=dict(width=0, color=band), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows[active + "_lower_m3_h"], mode="lines", line=dict(width=0, color=band), fill="tonexty", fillcolor=band, name="Nominal 90% interval", hoverinfo="skip"))
    hist = pd.DataFrame(result["history_chart"])
    fig.add_trace(go.Scatter(x=list(range(-len(hist) + 1, 1)), y=hist.biogas_m3_h, line=dict(color=SOFT8[0], width=2.5), name="Observed history"))
    fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows.past_m3_h, mode="lines+markers", line=dict(color=SOFT8[0], width=2.5, dash="dash"), name="History only"))
    if has_plan:
        fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows.feed_m3_h, mode="lines+markers", line=dict(color=SOFT8[1], width=2.5), name="With supplied plan"))
    fig.add_trace(go.Scatter(x=[0, 24], y=[rows.persistence_m3_h.iloc[0]] * 2, line=dict(color=SOFT8[2], width=1.5, dash="dot"), name="Persistence"))
    fig.add_vline(x=0, line_color=INK, line_dash="dash", line_width=1)
    fig.update_layout(
        template="plotly_white", height=370, margin=dict(l=20, r=20, t=58, b=18),
        font=dict(family="Arial", color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=11)),
        xaxis=dict(title="Hours from forecast origin", tickvals=[-24, -12, 0, 6, 12, 18, 24], ticktext=["−24", "−12", "Origin", "6", "12", "18", "24"], gridcolor=SOFT8[4], zeroline=False),
        yaxis=dict(title="Biogas flow (m³ h⁻¹)", gridcolor=SOFT8[4], zeroline=False),
        hovermode="x unified", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    return fig


metadata = bundle()["metadata"]
with st.sidebar:
    st.subheader("Muscatine WRRF")
    st.caption("Hourly metered gas flow to the boilers and burner.")
    st.markdown("**Fitted model**  \nPersistence-anchored LightGBM")
    st.markdown("**Forecast horizons**  \n1 · 2 · 3 · 4 · 6 · 9 · 12 · 18 · 24 h")
    st.divider()
    st.markdown("**Input files**")
    st.download_button("History example · CSV", (ROOT / "data" / "history_168h.csv").read_bytes(), "history_168h.csv", "text/csv", width="stretch")
    st.download_button("Recorded-feed example · CSV", (ROOT / "data" / "recorded_feed_oracle_24h.csv").read_bytes(), "recorded_feed_oracle_24h.csv", "text/csv", width="stretch")
    st.caption("The recorded-feed file contains executed feeding used as an oracle, rather than an issued plan.")
    st.divider()
    st.caption("Research prototype · single-site validation")
    st.caption("Uploaded files are processed on the host. The app does not save them to disk.")
    st.markdown("[Code and models](https://github.com/17609858895/Muscatine-Biogas-Streamlit)")
    st.caption(f"Version {VERSION}")

st.markdown('<div class="intro"><h1>Biogas forecast</h1><p>Use process history and a feeding plan to compare hourly to day-ahead predictions.</p></div>', unsafe_allow_html=True)
forecast_tab, batch_tab, model_tab = st.tabs(["Forecast", "Batch forecasts", "Model validation"])

with forecast_tab:
    history, plan, origin = None, None, None
    input_error = None
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.subheader("1 · Process history")
        source = st.radio("History source", ["Historical example", "Upload file"], horizontal=True)
        if source == "Historical example":
            history = example_history()
            st.caption("168 hourly observations · held-out historical example")
        else:
            upload = st.file_uploader("History CSV or XLSX", type=["csv", "xlsx"], key="history_upload")
            if upload is not None:
                try:
                    history = inference.uploaded_frame(upload.getvalue(), upload.name)
                except ValueError as exc:
                    input_error = str(exc)
        if history is not None:
            try:
                origin = inference.validate_history_frame(history)
                st.caption(f"Forecast origin: {origin:%Y-%m-%d %H:%M} · {len(history):,} input rows")
            except (ValueError, KeyError) as exc:
                input_error = str(exc)
        st.caption("At least 168 consecutive hourly rows; the last row sets the origin. Missing values are rejected.")
        with st.expander("Required columns and units"):
            st.markdown("`timestamp` · local time, on the hour\n\nGas and liquid flows: m³ h⁻¹; cover/tank heights: m; temperatures: °C; previous-day volume: m³; PS on-time: 0–1.")
            st.code("\n".join(inference.HISTORY_COLUMNS[1:]), language=None)
        if history is not None:
            with st.expander("Preview process history"):
                st.dataframe(history.tail(12), hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.subheader("2 · Feeding plan")
        include_plan = st.checkbox("Compare with a 24-hour feeding plan")
        if include_plan:
            plan_source = st.selectbox("Plan source", ["Create a plan", "Upload file", "Recorded-feed example (oracle)"])
            if plan_source == "Create a plan" and origin is not None:
                defaults = example_plan()
                a, b, c = st.columns(3)
                hsw = a.number_input("HSW (m³ h⁻¹)", min_value=0.0, value=float(defaults.hsw_m3_h.mean()), step=0.1)
                ps = b.number_input("PS on-time (0–1)", min_value=0.0, max_value=1.0, value=float(defaults.ps_on_fraction.mean()), step=0.01)
                twas = c.number_input("TWAS (m³ h⁻¹)", min_value=0.0, value=float(defaults.twas_m3_h.mean()), step=0.1)
                plan = inference.constant_plan(origin, hsw, ps, twas)
                st.caption("Entered values apply to each hour. Expand the table to edit individual hours.")
            elif plan_source == "Upload file":
                p_upload = st.file_uploader("Plan CSV or XLSX", type=["csv", "xlsx"], key="plan_upload")
                if p_upload is not None:
                    try:
                        plan = inference.uploaded_frame(p_upload.getvalue(), p_upload.name)
                        missing = set(inference.PLAN_COLUMNS) - set(plan)
                        if missing:
                            raise ValueError("Missing plan columns: " + ", ".join(sorted(missing)))
                    except ValueError as exc:
                        input_error = str(exc)
            elif plan_source == "Recorded-feed example (oracle)":
                plan = example_plan()
                st.caption("Executed future feeding from the example period. This is oracle information.")
            if plan is not None and set(inference.PLAN_COLUMNS).issubset(plan):
                with st.expander("Edit individual plan hours"):
                    editor_key = "plan_" + hashlib.sha256(inference.frame_csv(plan).encode()).hexdigest()[:12]
                    plan = st.data_editor(plan, hide_index=True, width="stretch", height=200, disabled=["timestamp"], key=editor_key,
                        column_config={
                            "hsw_m3_h": st.column_config.NumberColumn("HSW (m³ h⁻¹)", min_value=0.0),
                            "ps_on_fraction": st.column_config.NumberColumn("PS on-time", min_value=0.0, max_value=1.0),
                            "twas_m3_h": st.column_config.NumberColumn("TWAS (m³ h⁻¹)", min_value=0.0),
                        })
                multiplier = st.selectbox("HSW scenario", ["100% of supplied HSW", "75% of supplied HSW", "125% of supplied HSW"])
                plan = plan.copy()
                try:
                    plan["hsw_m3_h"] = pd.to_numeric(plan.hsw_m3_h, errors="raise") * {"100%": 1.0, "75%": .75, "125%": 1.25}[multiplier.split()[0]]
                except ValueError:
                    input_error = "The HSW plan must contain numeric values."
        else:
            st.caption("History-only forecasts use past process measurements and target-time calendar features.")
        st.caption("A plan must contain 24 hourly rows beginning one hour after the forecast origin. HSW is the total of both feed lines.")
    ready = history is not None and origin is not None and input_error is None and (not include_plan or plan is not None)
    if input_error:
        st.error(input_error)
    run = st.button("Generate forecast", type="primary", width="stretch", disabled=not ready, key="generate_forecast")
    if ready and (run or ("forecast_result" not in st.session_state and source == "Historical example" and not include_plan)):
        try:
            with st.spinner("Forecasting…"):
                result = inference.forecast(history, plan)
            st.session_state.forecast_result = {"result": result, "history": history.copy(), "plan": None if plan is None else plan.copy(), "signature": signature(history, plan)}
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if "forecast_result" in st.session_state:
        saved = st.session_state.forecast_result
        result = saved["result"]
        if not ready or signature(history, plan) != saved["signature"]:
            st.info("Inputs changed. Generate a forecast to update the displayed results.")
        frame = inference.result_frame(result)
        active = "feed" if "feed_m3_h" in frame else "past"
        st.caption(f"Forecast origin · {result['origin']} · {result['mode']}")
        cols = st.columns(3)
        for col, h in zip(cols, [1, 6, 24]):
            row = frame[frame.horizon_h == h].iloc[0]
            col.metric(f"{h} h ahead (m³ h⁻¹)", f"{row[active + '_m3_h']:.1f}")
        with st.container(border=True):
            st.subheader("Hourly to day-ahead outlook")
            st.plotly_chart(chart(result), width="stretch", config={"displaylogo": False}, key="forecast_chart")
            st.caption("Nine discrete forecast horizons. The lines connect model outputs; the shaded band is the selected model's validation-calibrated 90% interval.")
        if result["out_of_training_range"]:
            st.warning("History values outside the training range: " + ", ".join(result["out_of_training_range"]))
        st.caption("Plan differences are conditional predictions. The interface has not been validated in plant operation. Its LightGBM model and fixed intervals differ from the manuscript ensemble and online ACI.")
        with st.expander("Forecast values and intervals", expanded=True):
            st.dataframe(frame, hide_index=True, width="stretch", height=350)
        c1, c2 = st.columns(2)
        c1.download_button("Download forecasts · CSV", frame.to_csv(index=False).encode("utf-8-sig"), "muscatine_forecasts.csv", "text/csv", width="stretch")
        c2.download_button("Download forecasts and inputs · XLSX", inference.workbook_bytes(frame, saved["history"], saved["plan"]), "muscatine_forecasts.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

with batch_tab:
    st.subheader("Forecast consecutive origins")
    st.write("Upload a continuous hourly history. Each origin uses its preceding 168 hours and produces the nine history-only forecasts.")
    batch_source = st.radio("Batch source", ["Historical batch example", "Upload file"], horizontal=True)
    batch_history = None
    if batch_source == "Historical batch example":
        batch_history = example_batch()
    else:
        batch_upload = st.file_uploader("Batch history CSV or XLSX", type=["csv", "xlsx"], key="batch_upload")
        if batch_upload is not None:
            try:
                batch_history = inference.uploaded_frame(batch_upload.getvalue(), batch_upload.name)
            except ValueError as exc:
                st.error(str(exc))
    if batch_history is not None:
        try:
            inference.validate_history_frame(batch_history)
            count = min(100, len(batch_history) - 167)
            n_origins = st.number_input("Number of most recent origins", min_value=1, max_value=count, value=min(10, count), step=1)
            st.caption(f"{count} origins available, up to the 100-origin limit. A single 168-row example supplies one origin.")
            if st.button("Run batch forecasts", type="primary", width="stretch", key="run_batch"):
                with st.spinner("Forecasting each origin…"):
                    output = inference.batch_forecast(batch_history, n_origins)
                st.session_state.batch_result = {"output": output, "history": batch_history.copy()}
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if "batch_result" in st.session_state:
        saved = st.session_state.batch_result
        output = saved["output"]
        st.caption(f"{output.origin.nunique()} origins · {len(output):,} forecast rows · history-only model")
        st.dataframe(output, hide_index=True, width="stretch", height=350)
        b1, b2 = st.columns(2)
        b1.download_button("Download batch · CSV", output.to_csv(index=False).encode("utf-8-sig"), "muscatine_batch.csv", "text/csv", width="stretch")
        b2.download_button("Download batch and inputs · XLSX", inference.workbook_bytes(output, saved["history"]), "muscatine_batch.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

with model_tab:
    st.subheader("Hold-out evaluation of the GUI model")
    m1, m2, m3 = st.columns(3)
    m1.metric("Training origins", f"{metadata['training_rows']:,}")
    m2.metric("Validation origins", f"{metadata['validation_rows']:,}")
    m3.metric("Test origins per horizon", f"{metadata['test_rows_per_horizon']:,}")
    metrics = pd.read_csv(ROOT / "validation" / "gui_metrics_with_mae.csv")
    fig = go.Figure()
    for regime, label, color in [("past", "History only", SOFT8[0]), ("feed", "Executed-feed oracle", SOFT8[1])]:
        subset = metrics[metrics.regime == regime]
        fig.add_trace(go.Scatter(x=subset.h, y=subset.test_R2, mode="lines+markers", name=label, line=dict(color=color, width=2.5)))
    fig.update_layout(template="plotly_white", height=330, font=dict(color=INK, family="Arial"), legend=dict(orientation="h", y=1.1), margin=dict(l=20,r=20,t=50,b=20), xaxis=dict(title="Forecast horizon (h)", tickvals=[1,6,12,18,24], gridcolor=SOFT8[4]), yaxis=dict(title="Test R²", range=[0,1], gridcolor=SOFT8[4]))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="validation_chart")
    st.caption("The future-feed hold-out results use executed feeding as an oracle. Performance with issued operator plans was not tested.")
    display = metrics[["h", "regime", "test_R2", "test_RMSE_m3_h", "test_MAE_m3_h", "coverage"]].copy()
    display["regime"] = display.regime.map({"past": "History only", "feed": "Executed-feed oracle"})
    display.columns = ["Horizon (h)", "Information", "Test R²", "RMSE (m³ h⁻¹)", "MAE (m³ h⁻¹)", "90% interval coverage"]
    st.dataframe(display, hide_index=True, width="stretch", height=350)
    st.write("Intervals are calibrated separately for each horizon and information version using absolute errors from 696 validation origins. They remain fixed during app use; negative lower bounds are retained.")
    st.caption("Target: metered gas flow, rather than biological gas production. Training and validation cover one facility; external-site transfer and live operation have not been evaluated.")
    with st.expander("Model and input metadata"):
        st.json(metadata, expanded=False)
