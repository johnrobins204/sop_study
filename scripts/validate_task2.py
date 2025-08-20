# Small runtime validator for task 2 acceptance criteria.
import sys
from pathlib import Path

# ensure repo root on PYTHONPATH when run from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import get_model_instance, LanguageModel  # type: ignore
from src.types import ModelResponse  # type: ignore

def run():
    m = get_model_instance("google:test-model", config={"foo": "bar"}, api_params={"temp": 0.1})
    assert isinstance(m, LanguageModel), "get_model_instance did not return LanguageModel"
    r = m.generate("hello world")
    assert isinstance(r, ModelResponse), "LanguageModel.generate did not return ModelResponse"

    m2 = get_model_instance("ollama:demo")
    assert isinstance(m2, LanguageModel)
    r2 = m2.generate("prompt 2")
    assert isinstance(r2, ModelResponse)

    print("task 2 validation OK")

if __name__ == "__main__":
    run()