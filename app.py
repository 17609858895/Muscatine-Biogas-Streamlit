from pathlib import Path
from html import escape
import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import inference

ROOT = Path(__file__).resolve().parent
SOFT8 = ["#79C1E4", "#E68282", "#B2D362", "#BAE1F3", "#D4EAF8", "#EECDD5", "#F8E6E4", "#D1E4A6"]
INK = "#30343B"
VERSION = "2026.10.06.3"

st.set_page_config(page_title="Muscatine | Biogas forecast", layout="wide")
st.markdown("""

<style>
.stApp { background: #F6F8F7; color: #24323F; font-family: "Segoe UI", Arial, sans-serif; }
.stMainBlockContainer { max-width: 1320px; padding: 1.4rem 2rem 2.4rem; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { color: #24323F; letter-spacing: 0; }
h3 { font-size: 1.13rem !important; font-weight: 700; padding: 0 0 .4rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.5; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #526575 !important; font-size: 15px !important; line-height: 1.45; opacity: 1;
}
[data-testid="stWidgetLabel"] p { font-size: 16px; font-weight: 600; color: #24323F; }
.app-header {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    padding: 21px 25px; margin-bottom: 8px; border-radius: 11px;
    background: linear-gradient(115deg, #173F45 0%, #2F6F73 63%, #5EA7A3 100%);
    box-shadow: 0 8px 22px rgba(24,66,72,.11);
}
.brand h1 { color: #FFFFFF !important; margin: 0; padding: 0; font-size: 31px; font-weight: 750; line-height: 1.2; }
.brand p { color: #F0FAFA !important; margin: 9px 0 0; font-size: 16px; line-height: 1.45; }
.header-tags { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }
.header-tags span {
    color: #FFFFFF; font-size: 14px; font-weight: 600; white-space: nowrap;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.28);
    border-radius: 6px; padding: 5px 10px;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 6px; padding: 0 0 7px; border-bottom: 1px solid #D8E3E0; background: transparent; margin-bottom: 20px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 6px 6px 0 0; padding: 9px 18px; min-height: 43px; color: #526575;
}
[data-testid="stTabs"] [role="tab"] p { font-size: 16px; font-weight: 650; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #20585D; background: #E5F1EF; }
[data-testid="stTabs"] .react-aria-SelectionIndicator { background: #2F6F73; height: 2px; }
.st-key-input_panel, .st-key-result_panel, .st-key-batch_input_panel,
.st-key-batch_result_panel, .st-key-validation_header,
.st-key-validation_plot_panel, .st-key-validation_table_panel {
    background: #FFFFFF; border: 1px solid #DEE7E4 !important; border-radius: 10px;
    padding: 20px 22px; gap: 12px; box-shadow: 0 5px 18px rgba(36,50,63,.035);
}
.st-key-forecast_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
.st-key-batch_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
    flex-direction: column; gap: 20px;
}
.st-key-forecast_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-batch_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width: 100%; min-width: 0; flex: 1 1 100%;
}
.st-key-input_sections > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
    flex-direction: row; gap: 28px;
}
.st-key-input_sections > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width: calc(50% - 14px); min-width: 0; flex: 1 1 calc(50% - 14px);
}
.st-key-input_panel h3 {
    color: #2F6F73; font-size: 1.06rem !important; padding-bottom: 10px;
    border-bottom: 1px solid #E3ECE9; margin-bottom: 4px;
}
.st-key-input_panel [data-testid="stVerticalBlock"] { gap: 10px; }
.st-key-input_panel hr { margin: 2px 0; }
[data-testid="stMetric"] {
    background: #F8FBFC; border: 1px solid #E0E9E9; border-left: 4px solid #79C1E4;
    border-radius: 8px; padding: 14px 16px; min-height: 104px;
}
.st-key-forecast_metrics [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"],
.st-key-validation_metrics [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { border-left-color: #B2D362; }
.st-key-forecast_metrics [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"],
.st-key-validation_metrics [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { border-left-color: #E68282; }
[data-testid="stMetricLabel"] p { color: #526575; font-size: 15px; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 2.05rem; font-weight: 750; color: #24323F; }
.st-key-forecast_metrics { margin: 4px 0 2px; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    border: 1px solid #CADBD6; border-radius: 7px; min-height: 43px; color: #24323F; background: #FFFFFF;
}
[data-testid="stButton"] button p, [data-testid="stDownloadButton"] button p { font-size: 16px; font-weight: 650; }
[data-testid="stButton"] button[kind="primary"] {
    color: #FFFFFF !important; background: #2F6F73; border-color: #2F6F73; min-height: 46px;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
    border-color: #2F6F73; background: #EDF5F3; color: #20585D;
}
[data-testid="stButton"] button[kind="primary"]:hover { color: #FFFFFF !important; background: #245B60; }
[data-testid="stButton"] button:disabled { color: #637588 !important; background: #E1EAF2; border-color: #D6E0E6; }
button:focus-visible, input:focus-visible, [role="tab"]:focus-visible { outline: 2px solid #2F6F73; outline-offset: 3px; }
[data-testid="stExpander"] { background: transparent; }
[data-testid="stExpander"] details { border-color: #DCE6E2; border-radius: 7px; }
[data-testid="stExpander"] summary p { font-size: 15px; font-weight: 550; color: #526575; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border-radius: 7px; overflow: hidden; }
[data-testid="stDivider"] { margin: 0; }
.panel-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.panel-heading h2 { color: #24323F; font-size: 22px; font-weight: 750; line-height: 1.3; margin: 0; padding: 0; }
.mode-badge { color: #20585D; font-size: 14px; font-weight: 650; background: #E5F1EF; border-radius: 6px; padding: 6px 10px; white-space: nowrap; }
.empty-state { padding: 36px 16px 40px; text-align: center; color: #526575; }
.empty-state svg { margin-bottom: 16px; }
.empty-state h3 { margin: 0 0 8px; font-size: 20px; }
.empty-state p { margin: 0; font-size: 16px; }
.app-footer {
    display: flex; align-items: center; justify-content: space-between; gap: 20px;
    margin-top: 20px; padding-top: 16px; border-top: 1px solid #D8E3E0; font-size: 13px; color: #526575;
}
.app-footer a { color: #526575; text-underline-offset: 3px; }
@media (max-width: 1100px) {
    .stMainBlockContainer { padding: 1.2rem 1.5rem 2rem; }
    .st-key-validation_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 20px; }
    .st-key-validation_workspace > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100%; min-width: 0; flex: 1 1 100%; }
}
@media (max-width: 640px) {
    .stMainBlockContainer { padding: 1rem; }
    .app-header { padding: 20px; }
    .brand h1 { font-size: 25px; }
    .brand p { font-size: 15px; }
    .header-tags { display: none; }
    .st-key-input_panel, .st-key-result_panel, .st-key-batch_input_panel,
    .st-key-batch_result_panel, .st-key-validation_header,
    .st-key-validation_plot_panel, .st-key-validation_table_panel { padding: 17px; }
    .st-key-input_sections > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { flex-direction: column; gap: 18px; }
    .st-key-input_sections > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { width: 100%; flex: 1 1 100%; }
    [data-testid="stTabs"] [role="tab"] { padding: 8px 10px; }
    [data-testid="stMetric"] { padding: 12px 10px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .app-footer { flex-direction: column; align-items: flex-start; gap: 4px; }
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
    fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows[active + "_lower_m3_h"], mode="lines", line=dict(width=0, color=band), fill="tonexty", fillcolor=band, name="90% interval", legendrank=5, hoverinfo="skip"))
    hist = pd.DataFrame(result["history_chart"])
    fig.add_trace(go.Scatter(x=list(range(-len(hist) + 1, 1)), y=hist.biogas_m3_h, line=dict(color=SOFT8[0], width=2.5), name="Observed history", legendrank=1))
    fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows.past_m3_h, mode="lines+markers", line=dict(color=SOFT8[0], width=2.5, dash="dash"), name="History only", legendrank=2))
    if has_plan:
        fig.add_trace(go.Scatter(x=rows.horizon_h, y=rows.feed_m3_h, mode="lines+markers", line=dict(color=SOFT8[1], width=2.5), name="With supplied plan", legendrank=3))
    fig.add_trace(go.Scatter(x=[0, 24], y=[rows.persistence_m3_h.iloc[0]] * 2, line=dict(color=SOFT8[2], width=1.5, dash="dot"), name="Persistence", legendrank=4))
    fig.add_vline(x=0, line_color=INK, line_dash="dash", line_width=1)
    fig.update_layout(
        template="plotly_white", height=390, margin=dict(l=24, r=12, t=50, b=32),
        font=dict(family="Arial", color=INK, size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0, font=dict(size=13), traceorder="normal"),
        xaxis=dict(title="Hours from forecast origin", tickfont=dict(size=13), tickvals=[-24, -12, 0, 6, 12, 18, 24], ticktext=["−24", "−12", "Origin", "6", "12", "18", "24"], gridcolor=SOFT8[4], zeroline=False),
        yaxis=dict(title="Biogas flow (m³ h⁻¹)", tickfont=dict(size=13), gridcolor=SOFT8[4], zeroline=False),
        hovermode="x unified", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    return fig


def panel_heading(title, badge=None):
    tag = '' if badge is None else f'<span class="mode-badge">{escape(badge)}</span>'
    st.markdown(f'<div class="panel-heading"><h2>{escape(title)}</h2>{tag}</div>', unsafe_allow_html=True)


metadata = bundle()["metadata"]
st.markdown('''
<header class="app-header">
  <div class="brand">
    <h1>Biogas Forecast for Muscatine WRRF</h1>
    <p>Hourly metered gas flow to boilers and burner · History and feeding-plan comparison</p>
  </div>
  <div class="header-tags"><span>1–24 h horizons</span><span>LightGBM · Research model</span></div>
</header>
''', unsafe_allow_html=True)
forecast_tab, batch_tab, model_tab = st.tabs(["Forecast", "Batch forecasts", "Model validation"])

with forecast_tab:
    history, plan, origin = None, None, None
    input_error = None
    with st.container(key="forecast_workspace"):
        input_col, result_col = st.columns([1, 2.25], gap="large")
    with input_col, st.container(border=True, key="input_panel"):
        panel_heading("Forecast inputs")
        with st.container(key="input_sections"):
            process_section, plan_section = st.columns(2, gap="large")
        with process_section:
            st.subheader("1 · Process history")
            source = st.radio("History source", ["Historical example", "Upload file"], horizontal=True, label_visibility="collapsed")
            if source == "Historical example":
                history = example_history()
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
                    label = "Historical example" if source == "Historical example" else "Uploaded history"
                    st.caption(f"{label} · {len(history):,} hourly records")
                except (ValueError, KeyError) as exc:
                    input_error = str(exc)
            with st.expander("Data format and example"):
                st.write("At least 168 consecutive hourly rows. The last row sets the forecast origin. Missing values are rejected.")
                st.markdown("`timestamp`: local time, on the hour. Gas and liquid flows: m³ h⁻¹; cover/tank heights: m; temperatures: °C; previous-day volume: m³; PS on-time: 0–1.")
                st.code("\n".join(inference.HISTORY_COLUMNS[1:]), language=None)
                st.download_button("History example · CSV", (ROOT / "data/history_168h.csv").read_bytes(), "history_168h.csv", "text/csv", width="stretch")
                if history is not None:
                    st.caption("Last 12 input rows")
                    st.dataframe(history.tail(12), hide_index=True, width="stretch", height=240, row_height=32)
        with plan_section:
            st.subheader("2 · Feeding plan")
            include_plan = st.checkbox("Compare with a 24-hour feeding plan")
            if include_plan:
                st.caption("Configure the 24-hour future-feed inputs below.")
            else:
                st.caption("Enable a plan to compare future-feed inputs with the history-only forecast.")
            with st.expander("Plan format and example"):
                st.write("Provide 24 hourly rows from origin + 1 h through origin + 24 h. HSW is the total of both feed lines.")
                st.code("\n".join(inference.PLAN_COLUMNS), language=None)
                st.download_button("Recorded-feed example · CSV", (ROOT / "data/recorded_feed_oracle_24h.csv").read_bytes(), "recorded_feed_oracle_24h.csv", "text/csv", width="stretch")
                st.caption("This file contains executed feeding used as an oracle, rather than an issued plan.")
        if include_plan:
            with st.container(key="plan_controls"):
                plan_source_col, scenario_col = st.columns(2, gap="large")
                plan_source = plan_source_col.selectbox("Plan source", ["Create a plan", "Upload file", "Recorded-feed example (oracle)"])
                if plan_source == "Create a plan" and origin is not None:
                    defaults = example_plan()
                    a, b, c = st.columns(3, gap="large")
                    hsw = a.number_input("HSW (m³ h⁻¹)", min_value=0.0, value=float(defaults.hsw_m3_h.mean()), step=0.1)
                    twas = b.number_input("TWAS (m³ h⁻¹)", min_value=0.0, value=float(defaults.twas_m3_h.mean()), step=0.1)
                    ps = c.number_input("PS on-time (0–1)", min_value=0.0, max_value=1.0, value=float(defaults.ps_on_fraction.mean()), step=0.01)
                    plan = inference.constant_plan(origin, hsw, ps, twas)
                    st.caption("These values apply to each hour; individual hours can be edited below.")
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
                        plan = st.data_editor(plan, hide_index=True, width="stretch", height=240, disabled=["timestamp"], key=editor_key,
                            column_config={
                                "hsw_m3_h": st.column_config.NumberColumn("HSW (m³ h⁻¹)", min_value=0.0),
                                "ps_on_fraction": st.column_config.NumberColumn("PS on-time", min_value=0.0, max_value=1.0),
                                "twas_m3_h": st.column_config.NumberColumn("TWAS (m³ h⁻¹)", min_value=0.0),
                            })
                    multiplier = scenario_col.selectbox("HSW scenario", ["100% of supplied HSW", "75% of supplied HSW", "125% of supplied HSW"])
                    plan = plan.copy()
                    try:
                        plan["hsw_m3_h"] = pd.to_numeric(plan.hsw_m3_h, errors="raise") * {"100%": 1.0, "75%": .75, "125%": 1.25}[multiplier.split()[0]]
                    except ValueError:
                        input_error = "The HSW plan must contain numeric values."
        st.divider()
        ready = history is not None and origin is not None and input_error is None and (not include_plan or plan is not None)
        if input_error:
            st.error(input_error)
        run = st.button("Generate forecast", type="primary", width="stretch", disabled=not ready, key="generate_forecast")
        st.caption("Past process measurements · 168 h history")
        if ready and (run or ("forecast_result" not in st.session_state and source == "Historical example" and not include_plan)):
            try:
                with st.spinner("Forecasting…"):
                    result = inference.forecast(history, plan)
                st.session_state.forecast_result = {"result": result, "history": history.copy(), "plan": None if plan is None else plan.copy(), "signature": signature(history, plan)}
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
    with result_col, st.container(border=True, key="result_panel"):
        if "forecast_result" in st.session_state:
            saved = st.session_state.forecast_result
            result = saved["result"]
            frame = inference.result_frame(result)
            active = "feed" if "feed_m3_h" in frame else "past"
            panel_heading("Forecast outlook", "With feeding inputs" if active == "feed" else "History only")
            st.caption(f"Origin {result['origin']} · Biogas flow (m³ h⁻¹)")
            if not ready or signature(history, plan) != saved["signature"]:
                st.info("Inputs changed. Generate a forecast to update the displayed results.")
            with st.container(key="forecast_metrics"):
                cols = st.columns(3)
                for col, h in zip(cols, [1, 6, 24]):
                    row = frame[frame.horizon_h == h].iloc[0]
                    col.metric(f"{h} h ahead", f"{row[active + '_m3_h']:.1f}")
            st.plotly_chart(chart(result), width="stretch", theme=None, config={"displaylogo": False}, key="forecast_chart")
            st.caption("Nine horizons · Validation-calibrated 90% interval")
            if result["out_of_training_range"]:
                st.warning("History values outside the training range: " + ", ".join(result["out_of_training_range"]))
            c1, c2 = st.columns(2)
            c1.download_button("Download forecasts · CSV", frame.to_csv(index=False).encode("utf-8-sig"), "muscatine_forecasts.csv", "text/csv", width="stretch")
            c2.download_button("Forecasts and inputs · XLSX", inference.workbook_bytes(frame, saved["history"], saved["plan"]), "muscatine_forecasts.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            with st.expander("Forecast values and intervals"):
                st.dataframe(frame, hide_index=True, width="stretch", height=380, row_height=36, column_config=forecast_columns())
        else:
            panel_heading("Forecast outlook")
            st.markdown('<div class="empty-state"><h3>Ready for your forecast</h3><p>Load process history, then select Generate forecast.</p></div>', unsafe_allow_html=True)
    with st.expander("Model scope and intervals"):
        st.write("This interface uses the persistence-anchored LightGBM model. It predicts hourly metered gas flow to the boilers and burner, rather than biological gas production.")
        st.write("Lines connect nine discrete model outputs. The selected model's nominal 90% interval is calibrated from 696 validation origins and remains fixed during app use. Negative lower bounds are retained. This differs from the manuscript ensemble and online ACI.")
        st.write("History-only forecasts use past process measurements and target-time calendar features. Plan differences are conditional predictions; issued-plan performance and plant operation have not been evaluated. Training and validation cover a single facility.")
        st.caption("Uploaded files are processed on the host. The app does not save them to disk. No live SCADA connection is made.")

with batch_tab:
    with st.container(key="batch_workspace"):
        batch_inputs, batch_results = st.columns([1, 2.25], gap="large")
    with batch_inputs, st.container(border=True, key="batch_input_panel"):
        panel_heading("Batch inputs")
        st.caption("Forecast consecutive origins from continuous hourly history.")
        batch_source = st.radio("Batch source", ["Historical batch example", "Upload file"], label_visibility="collapsed")
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
                st.caption(f"{count} available origins · Up to 100 per batch")
                if st.button("Run batch forecasts", type="primary", width="stretch", key="run_batch"):
                    with st.spinner("Forecasting each origin…"):
                        output = inference.batch_forecast(batch_history, n_origins)
                    st.session_state.batch_result = {"output": output, "history": batch_history.copy()}
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
        with st.expander("How batch forecasts work"):
            st.write("Each origin uses its preceding 168 hours and produces nine history-only forecasts. A 168-row input supplies one origin. Feeding plans are not used in batch mode.")
    with batch_results, st.container(border=True, key="batch_result_panel"):
        panel_heading("Batch results", "History only")
        if "batch_result" in st.session_state:
            saved = st.session_state.batch_result
            output = saved["output"]
            st.caption(f"{output.origin.nunique()} origins · {len(output):,} forecast rows · Biogas flow (m³ h⁻¹)")
            st.dataframe(output, hide_index=True, width="stretch", height=400, row_height=36, column_config=forecast_columns())
            b1, b2 = st.columns(2)
            b1.download_button("Download batch · CSV", output.to_csv(index=False).encode("utf-8-sig"), "muscatine_batch.csv", "text/csv", width="stretch")
            b2.download_button("Batch and inputs · XLSX", inference.workbook_bytes(output, saved["history"]), "muscatine_batch.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        else:
            st.markdown('''<div class="empty-state">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="#79C1E4" stroke-width="2" aria-hidden="true"><rect x="7" y="8" width="34" height="33" rx="5"/><path d="M7 18H41M16 5V12M32 5V12M15 25H21M27 25H33M15 33H21M27 33H33"/></svg>
              <h3>Run a batch to view results</h3><p>Select the number of recent origins, then run the forecast.</p>
            </div>''', unsafe_allow_html=True)

with model_tab:
    with st.container(border=True, key="validation_header"):
        panel_heading("Model validation", "Single-site hold-out")
        st.caption("Persistence-anchored LightGBM · 18 fitted models · Nine forecast horizons")
        with st.container(key="validation_metrics"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Training origins", f"{metadata['training_rows']:,}")
            m2.metric("Validation origins", f"{metadata['validation_rows']:,}")
            m3.metric("Test origins per horizon", f"{metadata['test_rows_per_horizon']:,}")
    metrics = pd.read_csv(ROOT / "validation/gui_metrics_with_mae.csv")
    with st.container(key="validation_workspace"):
        validation_plot, validation_table = st.columns([1.15, 1], gap="large")
    with validation_plot, st.container(border=True, key="validation_plot_panel"):
        panel_heading("Accuracy across horizons")
        st.caption("Test R² · History only and executed-feed oracle")
        fig = go.Figure()
        for regime, label, color in [("past", "History only", SOFT8[0]), ("feed", "Executed-feed oracle", SOFT8[1])]:
            subset = metrics[metrics.regime == regime]
            fig.add_trace(go.Scatter(x=subset.h, y=subset.test_R2, mode="lines+markers", name=label, line=dict(color=color, width=2.5)))
        fig.update_layout(template="plotly_white", height=440, font=dict(color=INK, family="Arial", size=14), legend=dict(orientation="h", y=1.1, font=dict(size=13)), margin=dict(l=20,r=12,t=56,b=32), xaxis=dict(title="Forecast horizon (h)", tickfont=dict(size=13), tickvals=[1,6,12,18,24], gridcolor=SOFT8[4]), yaxis=dict(title="Test R²", tickfont=dict(size=13), range=[0,1], gridcolor=SOFT8[4]), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False}, key="validation_chart")
    with validation_table, st.container(border=True, key="validation_table_panel"):
        panel_heading("Hold-out metrics")
        st.caption("RMSE and MAE in m³ h⁻¹ · Nominal 90% coverage")
        display = metrics[["h", "regime", "test_R2", "test_RMSE_m3_h", "test_MAE_m3_h", "coverage"]].copy()
        display["regime"] = display.regime.map({"past": "History only", "feed": "Executed-feed oracle"})
        display.columns = ["Hours", "Information", "R²", "RMSE", "MAE", "Coverage"]
        st.dataframe(display, hide_index=True, width="stretch", height=440, row_height=36, column_config={
            "Hours": st.column_config.NumberColumn(format="%d", width="small"),
            "Information": st.column_config.TextColumn(width="medium"),
            "R²": st.column_config.NumberColumn(format="%.3f", width="small"),
            "RMSE": st.column_config.NumberColumn(format="%.1f", width="small"),
            "MAE": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Coverage": st.column_config.NumberColumn(format="percent", width="small"),
        })
    st.caption("Future-feed results use executed feeding as an oracle. Performance with issued operator plans was not tested.")
    with st.expander("Validation details and model metadata"):
        st.write("Intervals are calibrated separately for each horizon and information version using absolute errors from 696 validation origins. They remain fixed during app use; negative lower bounds are retained.")
        st.write("Target: metered gas flow, rather than biological gas production. This compact GUI model differs from the manuscript NNLS ensemble. External-site transfer and live operation have not been evaluated.")
        st.json(metadata, expanded=False)

st.markdown(f'<footer class="app-footer"><span>Muscatine WRRF · Research model · Version {VERSION}</span><a href="https://github.com/17609858895/Muscatine-Biogas-Streamlit" target="_blank" rel="noopener noreferrer">Code and models ↗</a></footer>', unsafe_allow_html=True)
