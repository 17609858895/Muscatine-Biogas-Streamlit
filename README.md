# Muscatine biogas forecast

A Streamlit interface for the Muscatine Water Resource Recovery Facility study. The app reuses the fitted persistence-anchored LightGBM models from the local manuscript GUI, with their original feature order, unit conversions and validation calibration. It does not retrain a model at startup.

Live app: https://muscatine-biogas-forecast.streamlit.app/

Repository: https://github.com/17609858895/Muscatine-Biogas-Streamlit

## Use

- Load the bundled 168-hour historical example, or upload consecutive hourly CSV/XLSX data.
- Forecast hourly **metered gas flow to the boilers and burner**, in m³ h⁻¹, at 1, 2, 3, 4, 6, 9, 12, 18 and 24 hours.
- Optionally enter a 24-hour HSW/PS/TWAS feeding plan or upload one. Compare the history-only and plan-conditioned predictions.
- Edit individual plan hours, or compare 75%, 100% and 125% HSW scenarios.
- Run up to 100 consecutive history-only forecast origins in batch mode.
- Download CSV results or an XLSX workbook containing forecasts and the original input columns.

The default historical example is from the test period. The separately selectable recorded-feed example is **executed future feeding, used as an oracle**, and is not an issued operating plan. Changing a plan produces a conditional model forecast; it does not establish a causal intervention benefit.

Inputs must be complete, finite, unique and consecutive hourly records. Missing values are rejected rather than automatically filled. No live SCADA connection is made. Uploaded files are processed by the hosted Streamlit service; the application does not save them to disk.

## Model scope

The bundle includes 18 fitted models: nine history-only and nine feed-conditioned models. Every model predicts a flow change relative to the latest observed flow. There are 16 hourly inputs, eight lag bins covering the previous 168 hours, target-time calendar features, and three future-feed means in the plan-conditioned version.

GUI hold-out performance is reported in `validation/gui_model_validation.csv`: 1,797 test origins per horizon. At 24 h, history-only R² = 0.467 and RMSE = 58.17 m³ h⁻¹; executed-feed-oracle R² = 0.666 and RMSE = 46.03 m³ h⁻¹. These are metrics of the compact **GUI LightGBM**, not the manuscript NNLS ensemble. The model was fitted on 5,421 training origins and its nominal 90% symmetric intervals were calibrated on 696 validation origins. Interval lower bounds are retained, including negative values; this is a fixed validation-residual calibration, not the manuscript's online ACI.

This is a single-site research interface with retrospective validation. External transfer, issued-plan performance and plant operation have not been validated. It predicts metered gas use, not biological production or methane yield.

## Local setup

Use Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

| Setting | Value |
|---|---|
| Repository | `17609858895/Muscatine-Biogas-Streamlit` |
| Branch | `main` |
| Main file | `app.py` |
| Python (Advanced settings) | `3.12` |

Follow the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). `packages.txt` supplies Linux OpenMP support for LightGBM. No secrets are needed. The application is deployed at the live URL above. The repository contains its source and model artifacts.

## Reproduce the bundle

```bash
python scripts/export_bundle.py
python scripts/validate_app.py
```

`models/model_bundle.joblib` contains the fitted native boosters, feature schema, original calibration, input bounds, weights and model metadata. The original 18 LightGBM text files are retained as a portable source. `predictor.py` is copied unchanged from the local manuscript GUI. The native model and feature reconstruction fidelity records are retained in `validation/`.

The visual layout follows the author's existing N2O/MXene prediction apps: a deep-teal title banner, white workspaces on a pale background, clearly grouped inputs and independent forecast metric cards. Process-history and plan selectors sit side by side; the longer plan controls use full-width rows above the outlook chart and downloads. Data requirements and model details remain expandable. The validation curve and table stack below 1100 CSS pixels to show all six metric columns clearly. Tables display flows to one decimal place; CSV/XLSX exports retain the underlying precision. The quantitative charts use the exact **soft_8** base palette: `#79C1E4`, `#E68282`, `#B2D362`, `#BAE1F3`, `#D4EAF8`, `#EECDD5`, `#F8E6E4`, `#D1E4A6`.

Author: CHONG LIU. App version: 2026.10.06.3.
