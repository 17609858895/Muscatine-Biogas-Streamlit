"""File adapters and exports around the unchanged local GUI predictor."""
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import predictor

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 5 * 1024 * 1024
HISTORY_COLUMNS = ["timestamp", *[s[0] for s in predictor.SCHEMA]]
PLAN_COLUMNS = ["timestamp", "hsw_m3_h", "ps_on_fraction", "twas_m3_h"]


@lru_cache(maxsize=1)
def load_bundle():
    # Only this trusted, repository-owned bundle is ever deserialized.
    bundle = joblib.load(ROOT / "models" / "model_bundle.joblib")
    if len(bundle["models"]) != 18 or sum(bundle["weights"].values()) != 1.0:
        raise ValueError("The fitted model bundle is incomplete.")
    predictor.MODELS.update(bundle["models"])
    return bundle


def uploaded_frame(data: bytes, filename: str):
    if len(data) > MAX_BYTES:
        raise ValueError("The input exceeds the 5 MB file limit.")
    try:
        if filename.lower().endswith(".xlsx"):
            frame = pd.read_excel(BytesIO(data), engine="openpyxl")
        else:
            frame = pd.read_csv(BytesIO(data), float_precision="round_trip", encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError("Cannot read the file. Use UTF-8 CSV or an XLSX worksheet with one header row.") from exc
    if frame.empty:
        raise ValueError("The input file has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame


def frame_csv(frame):
    return frame.to_csv(index=False)


def forecast(history_frame, plan_frame=None):
    load_bundle()
    return predictor.predict(frame_csv(history_frame), None if plan_frame is None else frame_csv(plan_frame))


def validate_history_frame(frame):
    predictor.read_history(frame_csv(frame))
    return pd.Timestamp(frame["timestamp"].iloc[-1])


def constant_plan(origin, hsw, ps, twas):
    return pd.DataFrame({
        "timestamp": pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h"),
        "hsw_m3_h": hsw,
        "ps_on_fraction": ps,
        "twas_m3_h": twas,
    })


def batch_forecast(history_frame, origins=10):
    validate_history_frame(history_frame)
    if not 1 <= origins <= 100:
        raise ValueError("Choose between 1 and 100 forecast origins.")
    available = len(history_frame) - 167
    if origins > available:
        raise ValueError(f"Only {available} complete origins are available from {len(history_frame)} hourly rows.")
    results = []
    for end in range(len(history_frame) - origins + 1, len(history_frame) + 1):
        window = history_frame.iloc[end - 168:end]
        result = forecast(window)
        for row in result["rows"]:
            results.append({"origin": result["origin"], **row})
    return pd.DataFrame(results)


def workbook_bytes(results, history, plan=None):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Forecasts", index=False)
        history.to_excel(writer, sheet_name="History_input", index=False)
        if plan is not None:
            plan.to_excel(writer, sheet_name="Plan_input", index=False)
        props = writer.book.properties
        props.creator = "CHONG LIU"
        props.lastModifiedBy = "CHONG LIU"
        props.title = "Muscatine hourly biogas forecasts"
    return buf.getvalue()


def result_frame(result):
    return pd.DataFrame([{"origin": result["origin"], **row} for row in result["rows"]])
