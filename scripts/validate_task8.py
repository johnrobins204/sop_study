import sys
from pathlib import Path
import tempfile
import os
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import orchestrate
from src.logging_config import LOG_FILE
import pandas as pd

def run():
    tmpdir = Path(tempfile.gettempdir())
    # prepare input for inference
    in_csv = tmpdir / f"inference_in_{os.getpid()}.csv"
    df = pd.DataFrame([
        {"prompt": "Hello world", "model_id": "google:test"},
        {"prompt": "Short answer", "model_id": "ollama:demo"},
    ])
    df.to_csv(in_csv, index=False)

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    inf_out = artifacts_dir / f"inference_raw_{os.getpid()}.csv"
    judged_out = artifacts_dir / f"inference_judged_{os.getpid()}.csv"
    analysis_out = artifacts_dir / f"analysis_summary_{os.getpid()}.csv"

    # template dir for judge
    td = Path(tempfile.mkdtemp(prefix="judge_templates_"))
    (td / "positive_keywords.txt").write_text("good\nexcellent\ncorrect\n", encoding="utf-8")
    (td / "negative_keywords.txt").write_text("bad\nwrong\nincorrect\n", encoding="utf-8")

    cfg = {
        "steps": [
            {"name": "inference", "component": "inference", "config": {"input_csv": str(in_csv), "output_csv": str(inf_out), "default_model": "google:test"}},
            {"name": "judge", "component": "judge", "config": {"input_csv": str(inf_out), "output_csv": str(judged_out), "template_dir": str(td)}},
            {"name": "analyst", "component": "analyst", "config": {"input_csv": str(judged_out), "output_csv": str(analysis_out), "group_col": "model", "rating_col": "judge_rating"}},
        ]
    }

    cfg_file = tmpdir / f"run_config_{os.getpid()}.yaml"
    import yaml
    with cfg_file.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    try:
        # remove existing log to ensure we capture new events
        try:
            if LOG_FILE.exists():
                LOG_FILE.unlink()
        except Exception:
            pass

        res = orchestrate(str(cfg_file))
        assert res.get("success") is True, f"orchestrate failed: {res}"
        # check artifacts and provenance files exist
        for art in [inf_out, judged_out, analysis_out]:
            assert Path(art).exists(), f"artifact missing: {art}"
            meta = Path(str(art) + ".meta.json")
            assert meta.exists(), f"provenance missing for {art}"
        # check log file has been written and contains expected entries
        time.sleep(0.1)
        assert LOG_FILE.exists(), "log file missing"
        text = LOG_FILE.read_text(encoding="utf-8")
        assert "run_from_config start" in text or "Wrote dataframe" in text, "expected log entries not found"
    finally:
        # cleanup created files
        for p in [in_csv, inf_out, judged_out, analysis_out, cfg_file]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        try:
            for f in td.iterdir():
                f.unlink()
            td.rmdir()
        except Exception:
            pass

    print("task 8 validation OK")

if __name__ == "__main__":
    run()