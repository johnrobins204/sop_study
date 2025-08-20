from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class ModelResponse:
    model: str
    prompt: str
    completion: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ensure metadata is JSON-serializable
        if d.get("metadata") is None:
            d["metadata"] = {}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelResponse":
        return cls(
            model=d.get("model", ""),
            prompt=d.get("prompt", ""),
            completion=d.get("completion", ""),
            score=d.get("score"),
            metadata=d.get("metadata") or {},
        )


__all__ = ["ModelResponse"]