import sys
from pathlib import Path
from typing import List

import pandas as pd


def detect_rating_columns(df: pd.DataFrame) -> List[str]:
    """Return list of columns considered score dimensions (endswith '_rating')."""
    return [c for c in df.columns if c.strip().lower().endswith("_rating")]


def load_and_compute_averages(csv_path: Path, group_col: str = "ablation_condition") -> pd.DataFrame:
    """
    Load CSV, compute per-group averages for rating columns, print summary and return DataFrame.

    - csv_path: path to CSV file
    - group_col: column used to define study classes / groups
    Returns: grouped means DataFrame (groups x rating columns)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if group_col not in df.columns:
        raise KeyError(f"group column '{group_col}' not found in CSV columns: {list(df.columns)}")

    rating_cols = detect_rating_columns(df)
    if not rating_cols:
        raise ValueError("No rating columns found (columns ending with '_rating').")

    df[rating_cols] = df[rating_cols].apply(pd.to_numeric, errors="coerce")
    grouped = df.groupby(group_col)[rating_cols].mean().round(3)

    # Console output (kept human-readable)
    print(f"\nAverages of rating dimensions grouped by '{group_col}':\n")
    print(grouped.to_string())
    print("\nCounts per group (non-NaN values per dimension):\n")
    counts = df.groupby(group_col)[rating_cols].count()
    print(counts.to_string())

    return grouped