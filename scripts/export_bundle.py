"""Package the existing, fitted native models without retraining them.

Run from the repository root: python scripts/export_bundle.py
"""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import predictor


def main():
    metric_file = ROOT / "validation" / "gui_metrics_with_mae.csv"
    if not metric_file.exists():
        metric_file = ROOT / "validation" / "gui_model_validation.csv"
    metrics = pd.read_csv(metric_file)
    native = sorted((ROOT / "models").glob("*.txt"))
    models = {p.stem: lgb.Booster(model_str=p.read_text(encoding="utf-8")) for p in native}
    assert len(models) == 18
    assert set(models) == {f"{r}_{h}" for r in ("past", "feed") for h in predictor.HORIZONS}
    metadata = {
        "app_version": "2026.10.06",
        "site": "Muscatine Water Resource Recovery Facility",
        "model_type": "Persistence-anchored LightGBM (LGB-Delta)",
        "target_name": "Hourly metered biogas flow to the boilers and burner",
        "target_unit": "m3 h-1",
        "horizons_h": predictor.HORIZONS,
        "history_hours": 168,
        "training_rows": 5421,
        "validation_rows": 696,
        "test_rows_per_horizon": 1797,
        "weights": {"LightGBM": 1.0},
        "raw_columns": ["timestamp", *[s[0] for s in predictor.SCHEMA]],
        "plan_columns": ["timestamp", "hsw_m3_h", "ps_on_fraction", "twas_m3_h"],
        "engineered_columns": {k: m.feature_name() for k, m in models.items()},
        "history_schema": predictor.SCHEMA,
        "lag_bins": predictor.BINS,
        "missing_value_policy": "Reject incomplete or non-finite inputs. No automatic imputation.",
        "interval_method": "Fixed symmetric nominal 90% intervals calibrated separately on 696 validation origins",
        "test_metrics": metrics.to_dict(orient="records"),
        "caveats": [
            "This compact GUI model is distinct from the manuscript's NNLS ensemble and online ACI.",
            "Executed future feeds in the hold-out evaluation are oracle information, not issued operating plans.",
            "Plan changes yield conditional forecasts and do not establish intervention effects.",
            "Single-site retrospective validation; no external-site or live plant validation.",
            "Predictions are metered gas use, not biological gas production or methane yield.",
            "Fixed interval lower bounds can be negative and are not clipped.",
        ],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native},
    }
    bundle = {
        "models": models,
        "metadata": metadata,
        "calibration": json.loads((ROOT / "models" / "calibration.json").read_text(encoding="utf-8")),
        "history_bounds": json.loads((ROOT / "models" / "history_bounds.json").read_text(encoding="utf-8")),
        "weights": {"LightGBM": 1.0},
    }
    joblib.dump(bundle, ROOT / "models" / "model_bundle.joblib", compress=3)
    (ROOT / "models" / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"models": len(models), "bundle_bytes": (ROOT / "models" / "model_bundle.joblib").stat().st_size}))


if __name__ == "__main__":
    main()
