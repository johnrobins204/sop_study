from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time

import pandas as pd

from src.io import load_csv, write_dataframe
from src.logging_config import get_logger

_logger = get_logger("judge")


def _load_keywords(template_dir: Optional[str]) -> Dict[str, List[str]]:
    pos = []
    neg = []
    if not template_dir:
        return {"positive": pos, "negative": neg}
    td = Path(template_dir)
    if not td.exists():
        _logger.warning("template_dir does not exist: %s", template_dir)
        return {"positive": pos, "negative": neg}
    p_pos = td / "positive_keywords.txt"
    p_neg = td / "negative_keywords.txt"
    if p_pos.exists():
        pos = [l.strip() for l in p_pos.read_text(encoding="utf-8").splitlines() if l.strip()]
    if p_neg.exists():
        neg = [l.strip() for l in p_neg.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"positive": pos, "negative": neg}


def _score_completion(text: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
    text_low = (text or "").lower()
    pos_hits = [k for k in keywords.get("positive", []) if k.lower() in text_low]
    neg_hits = [k for k in keywords.get("negative", []) if k.lower() in text_low]

    if pos_hits and not neg_hits:
        rating = 5
    elif neg_hits and not pos_hits:
        rating = 1
    elif pos_hits and neg_hits:
        # mixed -> neutral middle score weighted by counts
        rating = int(round(3 + (len(pos_hits) - len(neg_hits)) * 0.5))
        rating = max(1, min(5, rating))
    else:
        # no keywords -> fallback heuristics
        if len(text_low.split()) >= 10:
            rating = 4
        elif len(text_low.split()) >= 3:
            rating = 3
        else:
            rating = 2

    justification = {
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "len_words": len(text_low.split())
    }
    return {"rating": rating, "justification": justification}


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _logger.info("run_from_config start for judge, keys=%s", list(cfg.keys()))
    """
    cfg keys:
      - input_csv: path to input CSV (required)
      - output_csv: path to write judged CSV (required)
      - template_dir: path to judge templates (required per acceptance; may be empty)
      - rating_col_pattern: optional, pattern to coerce existing rating-like columns (unused here)
    Returns {"success": bool, "artifacts": [output_csv], "error": optional}
    """
    required = ["input_csv", "output_csv", "template_dir"]
    for k in required:
        if k not in cfg:
            _logger.error("missing config key: %s", k)
            return {"success": False, "error": f"missing config key: {k}", "artifacts": []}

    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
    template_dir = cfg.get("template_dir")

    try:
        df = load_csv(input_csv)
    except Exception as e:
        _logger.exception("failed to load input_csv: %s", e)
        return {"success": False, "error": f"failed to load input_csv: {e}", "artifacts": []}

    # Coerce existing rating-like columns to numeric where present
    for col in list(df.columns):
        if col.lower().endswith("_rating") or col.lower() == "rating":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                _logger.warning("failed to coerce column %s to numeric", col)
                pass

    keywords = _load_keywords(template_dir)

    judge_ratings = []
    judge_justifs = []
    start_ts = time.time()
    for _, row in df.iterrows():
        completion = ""
        # prefer 'completion' column, else try common alternatives
        if "completion" in row and pd.notna(row["completion"]):
            completion = str(row["completion"])
        elif "response" in row and pd.notna(row["response"]):
            completion = str(row["response"])
        elif "answer" in row and pd.notna(row["answer"]):
            completion = str(row["answer"])
        else:
            completion = ""

        scored = _score_completion(completion, keywords)
        judge_ratings.append(scored["rating"])
        judge_justifs.append(json.dumps(scored["justification"], ensure_ascii=False))
    duration = time.time() - start_ts

    df["judge_rating"] = judge_ratings
    df["judge_justification"] = judge_justifs

    try:
        write_dataframe(df, output_csv, index=False)
        _logger.info("run_from_config judge finished, wrote %s (rows=%d) duration=%.2fs", output_csv, len(df), duration)
        return {"success": True, "artifacts": [output_csv]}
    except Exception as e:
        _logger.exception("failed to write output_csv: %s", e)
        return {"success": False, "error": f"failed to write output_csv: {e}", "artifacts": []}


def run(argv: Optional[List[str]] = None) -> int:
    """
    CLI wrapper. Args: --input, --output, --template-dir
    Returns 0 on success, non-zero on failure.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="judge")
    parser.add_argument("--input", required=True, help="Input CSV path (raw outputs)")
    parser.add_argument("--output", required=True, help="Output CSV path (judged)")
    parser.add_argument("--template-dir", required=True, help="Directory containing judge templates (positive_keywords.txt / negative_keywords.txt)")
    args = parser.parse_args(argv)
    cfg = {"input_csv": args.input, "output_csv": args.output, "template_dir": args.template_dir}
    res = run_from_config(cfg)
    return 0 if res.get("success") else 4