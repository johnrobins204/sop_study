import argparse
from typing import Dict, Any, List, Optional
import pandas as pd

from src.io import load_csv, write_dataframe  # uses centralized IO helpers
from src.logging_config import get_logger

_logger = get_logger("analyst")


def analyze_df(df: pd.DataFrame, group_col: str, rating_col: str = "rating") -> pd.DataFrame:
    _logger.info("analyze_df start group_col=%s rating_col=%s", group_col, rating_col)
    """
    Group by `group_col` and return a DataFrame with the mean of `rating_col`.
    Raises ValueError if rating_col or group_col are missing.
    """
    if group_col not in df.columns:
        _logger.error("group_col '%s' not found in DataFrame columns", group_col)
        raise ValueError(f"group_col '{group_col}' not found in DataFrame columns")
    if rating_col not in df.columns:
        _logger.error("rating_col '%s' not found in DataFrame columns", rating_col)
        raise ValueError(f"rating_col '{rating_col}' not found in DataFrame columns")
    grouped = df.groupby(group_col)[rating_col].mean().reset_index()
    grouped = grouped.rename(columns={rating_col: f"{rating_col}_mean"})
    _logger.info("analyze_df done (groups=%d)", len(grouped))
    return grouped


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _logger.info("run_from_config start for analyst with cfg keys=%s", list(cfg.keys()))
    """
    cfg must include:
      - input_csv: path to input CSV
      - output_csv: path to output CSV
      - group_col: column to group by
    optional:
      - rating_col: name of rating column (default 'rating')
    Returns: {"success": bool, "artifacts": [output_csv]}
    """
    required = ["input_csv", "output_csv", "group_col"]
    for k in required:
        if k not in cfg:
            _logger.error("missing config key: %s", k)
            return {"success": False, "error": f"missing config key: {k}", "artifacts": []}

    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
    group_col = cfg["group_col"]
    rating_col = cfg.get("rating_col", "rating")

    try:
        df = load_csv(input_csv)
        out_df = analyze_df(df, group_col=group_col, rating_col=rating_col)
        write_dataframe(out_df, output_csv, index=False)
        _logger.info("run_from_config finished successfully, wrote %s", output_csv)
        return {"success": True, "artifacts": [output_csv]}
    except Exception as e:
        _logger.exception("run_from_config failed: %s", e)
        return {"success": False, "error": str(e), "artifacts": []}


def run(argv: Optional[List[str]] = None) -> int:
    """
    CLI wrapper. Returns 0 on success, non-zero on failure.
    CLI args: --input, --output, --group-col, [--rating-col]
    """
    parser = argparse.ArgumentParser(prog="analyst")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--group-col", required=True, help="Column to group by")
    parser.add_argument("--rating-col", default="rating", help="Rating column (default 'rating')")

    args = parser.parse_args(argv)
    cfg = {
        "input_csv": args.input,
        "output_csv": args.output,
        "group_col": args.group_col,
        "rating_col": args.rating_col,
    }
    result = run_from_config(cfg)
    if result.get("success"):
        return 0
    else:
        return 2