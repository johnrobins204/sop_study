import sys
from pathlib import Path
import tempfile
import os
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import get_model_instance, LanguageModel
from src.types import ModelResponse

def run():
    # create a fake experiment module at 3_experiments/fake_experiment/src/models.py
    base = ROOT / "3_experiments" / "fake_experiment" / "src"
    base.mkdir(parents=True, exist_ok=True)
    mod_path = base / "models.py"
    mod_path.write_text(textwrap.dedent("""
    from src.models.impl import LanguageModel
    from src.types import ModelResponse

    class ExperimentModel(LanguageModel):
        def __init__(self, identifier, config=None):
            super().__init__(identifier, config)
        def generate(self, prompt, **kwargs):
            return ModelResponse(model=f"experiment:fake:{self.identifier}", prompt=prompt, completion="exp:"+prompt, score=0.5, metadata={})
    """), encoding="utf-8")

    try:
        inst = get_model_instance("experiment:fake_experiment:default", config={}, api_params={})
        assert isinstance(inst, LanguageModel)
        r = inst.generate("hello")
        assert isinstance(r, ModelResponse)
        print("task 7 validation OK")
    finally:
        # cleanup
        try:
            mod_path.unlink()
            # remove created directories if empty
            p = mod_path.parent
            while p != ROOT and p.name != "":
                try:
                    p.rmdir()
                except Exception:
                    break
                p = p.parent
        except Exception:
            pass


if __name__ == "__main__":
    run()