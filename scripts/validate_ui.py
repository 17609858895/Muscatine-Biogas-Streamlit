"""Exercise the displayed forecast values against the original model outputs."""
from pathlib import Path
import json
import sys

from streamlit.testing.v1 import AppTest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference


def main():
    history = pd.read_csv(ROOT / "data" / "history_168h.csv", float_precision="round_trip")
    plan = pd.read_csv(ROOT / "data" / "recorded_feed_oracle_24h.csv", float_precision="round_trip")
    expected = inference.forecast(history, plan)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) == 6
    default_24 = float(at.metric[2].value)
    assert default_24 == round(expected["rows"][-1]["past_m3_h"], 1)
    at.checkbox[0].check().run()
    assert not at.exception, [str(e) for e in at.exception]
    at.selectbox[0].select("Recorded-feed example (oracle)").run()
    assert not at.exception, [str(e) for e in at.exception]
    at.button(key="generate_forecast").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    oracle_24 = float(at.metric[2].value)
    assert oracle_24 == round(expected["rows"][-1]["feed_m3_h"], 1)
    at.button(key="run_batch").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.session_state["batch_result"]["output"]) == 90
    report = {"status": "passed", "default_24h_display": default_24, "oracle_24h_display": oracle_24, "uncaught_ui_errors": 0, "batch_rows": 90, "tested_modes": ["history only", "recorded-feed oracle", "10 consecutive forecast origins"]}
    (ROOT / "validation" / "streamlit_ui_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
