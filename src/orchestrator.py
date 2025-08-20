import importlib
from pathlib import Path
from typing import Dict, Any, List
import yaml

from src.io import write_provenance


_COMPONENT_MAP = {
    "inference": "src.inference",
    "judge": "src.judge",
    "analyst": "src.analyst",
}


def orchestrate(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config and execute listed steps sequentially.

    YAML expected shape:
      steps:
        - name: step1
          component: inference    # or explicit module: src.inference
          config: {...}
        - name: step2
          component: judge
          config: {...}

    Returns dict: {"success": bool, "artifacts": [...], "errors": [...]}
    """
    config_path = Path(config_path)
    if not config_path.exists():
        return {"success": False, "artifacts": [], "errors": [f"config not found: {config_path}"]}

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    steps = cfg.get("steps", [])
    artifacts: List[str] = []
    errors: List[str] = []

    for step in steps:
        name = step.get("name") or step.get("component") or "<unnamed>"
        comp = step.get("component")
        module_path = step.get("module") or _COMPONENT_MAP.get(comp)
        step_cfg = step.get("config", {})

        if not module_path:
            errors.append(f"{name}: unknown component and no module provided")
            break

        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            errors.append(f"{name}: failed to import module {module_path}: {e}")
            break

        if not hasattr(module, "run_from_config"):
            errors.append(f"{name}: module {module_path} missing run_from_config")
            break

        try:
            result = module.run_from_config(step_cfg)
        except Exception as e:
            errors.append(f"{name}: exception during run_from_config: {e}")
            break

        if not result.get("success"):
            errors.append(f"{name}: step failed: {result.get('error')}")
            break

        step_artifacts = result.get("artifacts", [])
        for art in step_artifacts:
            try:
                meta = write_provenance(art, {"step": name, "component": comp or module_path, "config": step_cfg})
            except Exception:
                meta = None
            artifacts.append(art)
            if meta:
                artifacts.append(meta)

    return {"success": len(errors) == 0, "artifacts": artifacts, "errors": errors}