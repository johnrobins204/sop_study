import sys
from pathlib import Path
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judge import run_from_config  # type: ignore
import pandas as pd

def run():
    # prepare input df
    df = pd.DataFrame([
        {"prompt": "p1", "completion": "This is correct and complete. Excellent answer."},
        {"prompt": "p2", "completion": "Wrong and misleading response."},
        {"prompt": "p3", "completion": "Short."},
    ])
    tmp_in = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    tmp_td = tempfile.TemporaryDirectory()
    try:
        df.to_csv(tmp_in.name, index=False)

        # create simple template dir files
        pos = Path(tmp_td.name) / "positive_keywords.txt"
        neg = Path(tmp_td.name) / "negative_keywords.txt"
        pos.write_text("correct\nexcellent\ncomplete\n", encoding="utf-8")
        neg.write_text("wrong\nmisleading\nincorrect\n", encoding="utf-8")

        tmp_out = str(Path(tempfile.gettempdir()) / f"judge_out_{os.getpid()}.csv")
        cfg = {"input_csv": tmp_in.name, "output_csv": tmp_out, "template_dir": tmp_td.name}
        res = run_from_config(cfg)
        assert res.get("success") is True, f"run_from_config failed: {res}"
        assert tmp_out in res.get("artifacts", []), "output artifact not reported"

        out_df = pd.read_csv(tmp_out)
        assert "judge_rating" in out_df.columns
        assert "judge_justification" in out_df.columns
        # numeric rating check
        assert pd.api.types.is_numeric_dtype(out_df["judge_rating"]), "judge_rating is not numeric"
        assert len(out_df) == 3
    finally:
        try:
            tmp_in.close()
            os.unlink(tmp_in.name)
        except Exception:
            pass
        try:
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)
        except Exception:
            pass
        tmp_td.cleanup()

    print("task 5 validation OK")

if __name__ == "__main__":
    run()