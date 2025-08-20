from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from pathlib import Path
from typing import Optional, Dict, Any

from ..impl import LanguageModel
from ...types import ModelResponse


def _experiment_models_path(exp_name: str) -> Path:
    # expects experiments under repo root at 3_experiments/<name>/src/models.py
    return Path.cwd() / "3_experiments" / exp_name / "src" / "models.py"


def load_experiment_module(exp_name: str):
    p = _experiment_models_path(exp_name)
    if not p.exists():
        raise FileNotFoundError(f"experiment models not found at {p}")
    loader = SourceFileLoader(f"experiment_models_{exp_name}", str(p))
    spec = spec_from_loader(loader.name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def get_experiment_model_instance(identifier: str, config: Optional[Dict[str, Any]] = None, api_params: Optional[Dict[str, Any]] = None) -> LanguageModel:
    """
    identifier format: "experiment:<exp_name>[:<model_id>]" or "experiment:<exp_name>"
    Adapter looks for 3_experiments/<exp_name>/src/models.py and attempts to:
      - call module.get_model_instance(model_id, config, api_params) if present, OR
      - call module.get_model(model_id, config) or module.ExperimentModel(...) heuristics.
    Returns a LanguageModel-compatible wrapper around the experiment model.
    """
    parts = identifier.split(":", 2)
    # parts[0] == 'experiment'
    if len(parts) < 2 or not parts[1]:
        raise ValueError("invalid experiment identifier, expected 'experiment:<name>[:<model_id>]'")
    exp_name = parts[1]
    model_id = parts[2] if len(parts) > 2 else None

    mod = load_experiment_module(exp_name)

    # Prefer an exported get_model_instance factory from the experiment module
    if hasattr(mod, "get_model_instance"):
        factory = getattr(mod, "get_model_instance")
        inst = factory(model_id or "", config or {}, api_params or {})
        # assume returned instance is already compatible
        return inst

    # If experiment module exposes a class named ExperimentModel or similar, attempt to wrap it
    if hasattr(mod, "ExperimentModel"):
        cls = getattr(mod, "ExperimentModel")
        inst = cls(model_id or "", config or {})
        # wrap to LanguageModel if necessary
        if isinstance(inst, LanguageModel):
            return inst

    # Fallback: try a function get_model(name, config)
    if hasattr(mod, "get_model"):
        inst = getattr(mod, "get_model")(model_id or "", config or {})
        if isinstance(inst, LanguageModel):
            return inst

    raise RuntimeError("experiment module did not expose a compatible model factory or class")