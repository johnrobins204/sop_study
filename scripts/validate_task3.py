import sys
from pathlib import Path
import tempfile
import csv
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analyst import analyze_df, run_from_config  # type: ignore
import pandas as pd

def run():
    # build sample dataframe
    df = pd.DataFrame([
        {"group": "A", "rating": 4.0},
        {"group": "A", "rating": 2.0},
        {"group": "B", "rating": 5.0},
        {"group": "B", "rating": 3.0},
    ])
    grouped = analyze_df(df, group_col="group", rating_col="rating")
    assert "rating_mean" in grouped.columns, "analyze_df did not produce rating_mean column"
    assert grouped.loc[grouped["group"] == "A", "rating_mean"].iloc[0] == 3.0

    # test run_from_config with real CSV files
    tmp_in = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    try:
        df.to_csv(tmp_in.name, index=False)
        tmp_out = str(Path(tempfile.gettempdir()) / f"analyst_out_{os.getpid()}.csv")
        cfg = {"input_csv": tmp_in.name, "output_csv": tmp_out, "group_col": "group", "rating_col": "rating"}
        res = run_from_config(cfg)
        assert res.get("success") is True, f"run_from_config failed: {res}"
        assert tmp_out in res.get("artifacts", []), "output artifact not reported"
        out_df = pd.read_csv(tmp_out)
        assert "rating_mean" in out_df.columns
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

    print("task 3 validation OK")

if __name__ == "__main__":
    run()