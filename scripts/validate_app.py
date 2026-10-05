"""Check prediction fidelity, input rejection, plan response and file exports."""
from io import BytesIO
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference
import predictor


def main():
    history = pd.read_csv(ROOT / "data" / "history_168h.csv", float_precision="round_trip")
    plan = pd.read_csv(ROOT / "data" / "recorded_feed_oracle_24h.csv", float_precision="round_trip")
    source_spec = importlib.util.spec_from_file_location("portable_native_predictor", ROOT / "predictor.py")
    source = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source)
    native = source.predict(
        (ROOT / "data" / "history_168h.csv").read_text(encoding="utf-8"),
        (ROOT / "data" / "recorded_feed_oracle_24h.csv").read_text(encoding="utf-8"),
    )
    fitted = inference.forecast(history, plan)
    a, b = pd.DataFrame(native["rows"]), pd.DataFrame(fitted["rows"])
    numeric = a.select_dtypes(include="number").columns
    parity = float(np.max(np.abs(a[numeric].to_numpy() - b[numeric].to_numpy())))
    assert parity < 1e-9
    bare = inference.forecast(history)
    no_plan = pd.DataFrame(bare["rows"])
    assert np.max(np.abs(no_plan.past_m3_h - b.past_m3_h)) == 0
    scaled = plan.copy()
    scaled["hsw_m3_h"] *= 1.25
    changed = pd.DataFrame(inference.forecast(history, scaled)["rows"])
    delta = float(changed.feed_m3_h.iloc[-1] - b.feed_m3_h.iloc[-1])
    assert abs(delta) > 0
    rejected = {}
    invalid = {
        "short_history": history.iloc[1:],
        "missing_column": history.drop(columns="cover_d1_m"),
        "nonfinite": history.assign(cover_d1_m=np.nan),
        "hourly_gap": history.drop(index=30),
        "negative_flow": history.assign(twas_m3_h=-1),
    }
    for name, frame in invalid.items():
        try:
            inference.forecast(frame)
        except ValueError as exc:
            rejected[name] = str(exc)
        else:
            raise AssertionError(f"Invalid history accepted: {name}")
    invalid_plan = plan.copy()
    invalid_plan["timestamp"] = pd.date_range("2020-01-01", periods=24, freq="h")
    try:
        inference.forecast(history, invalid_plan)
    except ValueError as exc:
        rejected["misaligned_plan"] = str(exc)
    else:
        raise AssertionError("Misaligned plan accepted")
    extra = history.assign(operator_note="retained input")
    result = inference.result_frame(fitted)
    workbook = inference.workbook_bytes(result, extra, plan)
    wb = openpyxl.load_workbook(BytesIO(workbook))
    assert wb.properties.creator == wb.properties.lastModifiedBy == "CHONG LIU"
    assert wb["Forecasts"].max_row == 10
    assert wb["History_input"].cell(1, len(extra.columns)).value == "operator_note"
    imported = inference.uploaded_frame(workbook, "forecasts.xlsx")
    assert len(imported) == 9
    # Reimport the input sheet independently, then verify the XLSX adapter with it.
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        history.to_excel(writer, index=False)
    xlsx_history = inference.uploaded_frame(buf.getvalue(), "history.xlsx")
    assert inference.validate_history_frame(xlsx_history) == pd.Timestamp(history.timestamp.iloc[-1])
    batch = inference.batch_forecast(history, 1)
    assert len(batch) == 9 and batch.origin.nunique() == 1
    assert np.max(np.abs(batch.past_m3_h - no_plan.past_m3_h)) == 0
    bundle = inference.load_bundle()
    assert len(bundle["models"]) == 18
    assert sum(bundle["weights"].values()) == 1.0
    assert all(bundle["metadata"]["engineered_columns"][k] == m.feature_name() for k,m in bundle["models"].items())
    report = {
        "status": "passed", "models": 18, "horizons": predictor.HORIZONS,
        "native_vs_bundle_max_abs_error_m3_h": parity,
        "history_only_unchanged_by_plan": True,
        "HSW_125_percent_change_24h_m3_h": delta,
        "rejected_inputs": rejected,
        "xlsx_export_metadata": {"creator": wb.properties.creator, "lastModifiedBy": wb.properties.lastModifiedBy},
        "xlsx_retains_extra_input_columns": True,
        "xlsx_history_adapter": "passed", "batch_rows": len(batch),
        "bundle_sha256": hashlib.sha256((ROOT / "models" / "model_bundle.joblib").read_bytes()).hexdigest(),
    }
    (ROOT / "validation" / "streamlit_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
