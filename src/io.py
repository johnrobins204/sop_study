import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
from datetime import datetime
from typing import Optional
import pandas as pd

from .logging_config import get_logger
_logger = get_logger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(p)


def write_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)
    _logger.info("Wrote dataframe to %s (rows=%d)", str(p), len(df))


def _git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def write_provenance(artifact_path: str, config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    meta = {
        "artifact": str(artifact_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "config": config,
    }
    if extra:
        meta["extra"] = extra
    # make deterministic meta filename: artifact + ".meta.json"
    meta_path = str(Path(artifact_path).with_name(Path(artifact_path).name + ".meta.json"))
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    _logger.info("Wrote provenance for %s -> %s", artifact_path, meta_path)
    return meta_path