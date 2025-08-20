import sys
from pathlib import Path
import tempfile
import os
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import run_from_config  # type: ignore
import pandas as pd

def run():
    tmp_dir = Path(tempfile.gettempdir())
    df = pd.DataFrame([
        {"prompt": "hello one", "model_id": "google:test"},
        {"prompt": "hello two", "model_id": "ollama:demo"},
    ])

    tmp_in = None
    tmp_out = tmp_dir / f"inference_out_{os.getpid()}.csv"
    try:
        # create a stable temp input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as fh:
            tmp_in = Path(fh.name)
            df.to_csv(tmp_in, index=False)

        cfg = {"input_csv": str(tmp_in), "output_csv": str(tmp_out), "default_model": "google:test"}
        res = run_from_config(cfg)
        assert res.get("success") is True, f"run_from_config failed: {res}"
        assert str(tmp_out) in res.get("artifacts", []), f"output artifact not reported, artifacts={res.get('artifacts')}"
        out_df = pd.read_csv(tmp_out)
        for col in ("model", "prompt", "completion", "metadata"):
            assert col in out_df.columns, f"missing column {col}"
        assert len(out_df) == 2
    except Exception as e:
        print("VALIDATION FAILED:", e)
        traceback.print_exc()
        raise
    finally:
        # cleanup
        try:
            if tmp_in and tmp_in.exists():
                tmp_in.unlink()
        except Exception:
            pass
        try:
            if tmp_out.exists():
                tmp_out.unlink()
        except Exception:
            pass

    print("task 4 validation OK")

if __name__ == "__main__":
    run()