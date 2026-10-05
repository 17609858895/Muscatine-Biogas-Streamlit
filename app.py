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
VERSION = "2026.10.06.1"

st.set_page_config(page_title="Muscatine | Biogas forecast", layout="wide")
st.markdown("""
<style>
.stApp { background: #F1F5FA; color: #30343B; }
.stMainBlockContainer { max-width: 1280px; padding-top: 2rem; padding-bottom: 2.5rem; }
[data-testid="stHeader"] { background: #F1F5FA; }
h1, h2, h3 { color: #263648; letter-spacing: -.3px; }
h3 { font-size: 1.2rem !important; padding-bottom: .3rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.6; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #43566A !important; font-size: 15px !important; line-height: 1.6; opacity: 1;
}
[data-testid="stWidgetLabel"] p { font-size: 15px; font-weight: 600; color: #30343B; }
[data-testid="stSidebar"] { background: #E8F0F7; border-right: 1px solid #CAD9E5; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1rem; }
.st-key-history_card, .st-key-plan_card, .st-key-outlook_card,
.st-key-batch_card, .st-key-validation_card {
    background: #FFFFFF; border: 1px solid #CAD9E5 !important;
    border-radius: 16px; padding: 22px; box-shadow: 0 3px 12px rgba(38, 54, 72, .04);
}
.st-key-history_card, .st-key-plan_card { border-top: 3px solid #79C1E4 !important; }
.st-key-plan_card { border-top-color: #E68282 !important; }
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #CAD9E5; border-radius: 14px;
    padding: 16px 20px; min-height: 116px; box-shadow: 0 3px 12px rgba(38, 54, 72, .04);
}
[data-testid="stMetricLabel"] p { color: #43566A; font-size: 15px; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 2.15rem; font-weight: 600; color: #263648; }
[data-testid="stTabs"] [role="tablist"] {
    gap: 8px; padding: 6px; border: 1px solid #CAD9E5; border-radius: 12px;
    background: #E8F0F7; margin-bottom: 18px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px; padding: 10px 20px; min-height: 44px; color: #43566A;
}
[data-testid="stTabs"] [role="tab"] p { font-size: 16px; font-weight: 600; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #263648; background: #FFFFFF; box-shadow: 0 2px 6px rgba(38, 54, 72, .06);
}
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    border: 1px solid #BACDDC; border-radius: 10px; min-height: 44px;
    color: #263648; background: #FFFFFF;
}
[data-testid="stButton"] button p, [data-testid="stDownloadButton"] button p { font-size: 15px; font-weight: 600; }
[data-testid="stButton"] button[kind="primary"] {
    color: #263648 !important; background: #79C1E4; border-color: #79C1E4;
    min-height: 48px; box-shadow: 0 3px 8px rgba(38, 54, 72, .08);
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
    border-color: #79C1E4; background: #EAF5FB; color: #263648;
}
[data-testid="stButton"] button[kind="primary"]:hover { background: #BAE1F3; }
[data-testid="stButton"] button:disabled { color: #637588 !important; background: #E1EAF2; }
button:focus-visible, input:focus-visible, [role="tab"]:focus-visible { outline: 2px solid #43566A; outline-offset: 3px; }
[data-testid="stExpander"] { background: #FFFFFF; border-radius: 10px; }
[data-testid="stExpander"] details { border-color: #CAD9E5; }
[data-testid="stExpander"] summary p { font-size: 15px; font-weight: 500; color: #30343B; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border-radius: 10px; overflow: hidden; }
.intro {
    background: #FFFFFF; border: 1px solid #CAD9E5; border-left: 5px solid #79C1E4;
    border-radius: 16px; padding: 24px 28px; margin: 0 0 22px;
    box-shadow: 0 3px 12px rgba(38, 54, 72, .04);
}
.intro h1 { margin: 0; padding: 0 0 8px; font-size: 2.1rem; line-height: 1.25; }
.intro p { margin: 0; font-size: 16px; line-height: 1.6; color: #43566A; }
.intro .eyebrow { margin-bottom: 8px; color: #43566A; font-size: 14px; font-weight: 600; }
@media (max-width: 640px) {
    .stMainBlockContainer { padding: 1rem; }
    .intro { padding: 20px; }
    .intro h1 { font-size: 1.8rem; }
    .st-key-history_card, .st-key-plan_card, .st-key-outlook_card,
    .st-key-batch_card, .st-key-validation_card { padding: 16px; }
    [data-testid="stTabs"] [role="tab"] { padding: 8px 12px; }
}
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


def forecast_columns():
    """Readable display labels and precision; downloaded values retain full precision."""
    columns = {
        "origin": st.column_config.TextColumn("Forecast origin"),
        "horizon_h": st.column_config.NumberColumn("Horizon (h)", format="%d"),
        "target_time": st.column_config.TextColumn("Forecast time"),
        "persistence_m3_h": st.column_config.NumberColumn("Persistence (m³ h⁻¹)", format="%.1f"),
    }
    for regime, label in [("past", "History"), ("feed", "Plan")]:
        for suffix, bound in [("_m3_h", ""), ("_lower_m3_h", " · lower"), ("_upper_m3_h", " · upper")]:
            columns[regime + suffix] = st.column_config.NumberColumn(label + bound + " (m³ h⁻¹)", format="%.1f")
    return columns


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
        template="plotly_white", height=400, margin=dict(l=20, r=20, t=66, b=24),
        font=dict(family="Arial", color=INK, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=13)),
        xaxis=dict(title="Hours from forecast origin", tickfont=dict(size=13), tickvals=[-24, -12, 0, 6, 12, 18, 24], ticktext=["−24", "−12", "Origin", "6", "12", "18", "24"], gridcolor=SOFT8[4], zeroline=False),
        yaxis=dict(title="Biogas flow (m³ h⁻¹)", tickfont=dict(size=13), gridcolor=SOFT8[4], zeroline=False),
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

st.markdown('<div class="intro"><p class="eyebrow">Muscatine WRRF · 1–24 h forecasts</p><h1>Biogas forecast</h1><p>Use process history and a feeding plan to compare hourly to day-ahead predictions.</p></div>', unsafe_allow_html=True)
forecast_tab, batch_tab, model_tab = st.tabs(["Forecast", "Batch forecasts", "Model validation"])

with forecast_tab:
    history, plan, origin = None, None, None
    input_error = None
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True, key="history_card", height="stretch"):
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
    with right, st.container(border=True, key="plan_card", height="stretch"):
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
        with st.container(border=True, key="outlook_card"):
            st.subheader("Hourly to day-ahead outlook")
            st.plotly_chart(chart(result), width="stretch", theme=None, config={"displaylogo": False}, key="forecast_chart")
            st.caption("Nine discrete forecast horizons. The lines connect model outputs; the shaded band is the selected model's validation-calibrated 90% interval.")
        if result["out_of_training_range"]:
            st.warning("History values outside the training range: " + ", ".join(result["out_of_training_range"]))
        st.caption("Plan differences are conditional predictions. The interface has not been validated in plant operation. Its LightGBM model and fixed intervals differ from the manuscript ensemble and online ACI.")
        with st.expander("Forecast values and intervals", expanded=True):
            st.dataframe(frame, hide_index=True, width="stretch", height=380, row_height=36, column_config=forecast_columns())
        c1, c2 = st.columns(2)
        c1.download_button("Download forecasts · CSV", frame.to_csv(index=False).encode("utf-8-sig"), "muscatine_forecasts.csv", "text/csv", width="stretch")
        c2.download_button("Download forecasts and inputs · XLSX", inference.workbook_bytes(frame, saved["history"], saved["plan"]), "muscatine_forecasts.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

with batch_tab, st.container(border=True, key="batch_card"):
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
        st.dataframe(output, hide_index=True, width="stretch", height=380, row_height=36, column_config=forecast_columns())
        b1, b2 = st.columns(2)
        b1.download_button("Download batch · CSV", output.to_csv(index=False).encode("utf-8-sig"), "muscatine_batch.csv", "text/csv", width="stretch")
        b2.download_button("Download batch and inputs · XLSX", inference.workbook_bytes(output, saved["history"]), "muscatine_batch.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

with model_tab, st.container(border=True, key="validation_card"):
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
    fig.update_layout(template="plotly_white", height=370, font=dict(color=INK, family="Arial", size=14), legend=dict(orientation="h", y=1.1, font=dict(size=13)), margin=dict(l=20,r=20,t=56,b=24), xaxis=dict(title="Forecast horizon (h)", tickfont=dict(size=13), tickvals=[1,6,12,18,24], gridcolor=SOFT8[4]), yaxis=dict(title="Test R²", tickfont=dict(size=13), range=[0,1], gridcolor=SOFT8[4]), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False}, key="validation_chart")
    st.caption("The future-feed hold-out results use executed feeding as an oracle. Performance with issued operator plans was not tested.")
    display = metrics[["h", "regime", "test_R2", "test_RMSE_m3_h", "test_MAE_m3_h", "coverage"]].copy()
    display["regime"] = display.regime.map({"past": "History only", "feed": "Executed-feed oracle"})
    display.columns = ["Horizon (h)", "Information", "Test R²", "RMSE (m³ h⁻¹)", "MAE (m³ h⁻¹)", "90% interval coverage"]
    st.dataframe(display, hide_index=True, width="stretch", height=380, row_height=36, column_config={
        "Horizon (h)": st.column_config.NumberColumn(format="%d"),
        "Test R²": st.column_config.NumberColumn(format="%.3f"),
        "RMSE (m³ h⁻¹)": st.column_config.NumberColumn(format="%.1f"),
        "MAE (m³ h⁻¹)": st.column_config.NumberColumn(format="%.1f"),
        "90% interval coverage": st.column_config.NumberColumn(format="percent"),
    })
    st.write("Intervals are calibrated separately for each horizon and information version using absolute errors from 696 validation origins. They remain fixed during app use; negative lower bounds are retained.")
    st.caption("Target: metered gas flow, rather than biological gas production. Training and validation cover one facility; external-site transfer and live operation have not been evaluated.")
    with st.expander("Model and input metadata"):
        st.json(metadata, expanded=False)
