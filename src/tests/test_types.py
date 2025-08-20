import json
from src.types import ModelResponse

def test_modelresponse_serialization():
    mr = ModelResponse(model="m1", prompt="p", completion="c", score=0.9, metadata={"k":"v"})
    d = mr.to_dict()
    assert d["model"] == "m1"
    assert d["prompt"] == "p"
    assert d["completion"] == "c"
    assert abs(d["score"] - 0.9) < 1e-6
    assert isinstance(d["metadata"], dict)
    # round-trip via from_dict
    mr2 = ModelResponse.from_dict(d)
    assert mr2.model == mr.model
    assert mr2.completion == mr.completion