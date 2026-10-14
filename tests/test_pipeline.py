import json
from types import SimpleNamespace

import pytest

from paper2code import PaperCoder, RefusedError, strip_fences

PLAN = {
    "overview": "Tiny MLP", "components": ["model"], "unclear_points": ["batch size"],
    "files": [{"path": "model.py", "purpose": "the model", "depends_on": []},
              {"path": "train.py", "purpose": "training", "depends_on": ["model.py"]}],
    "config_yaml": "lr: 0.001\n", "requirements": ["torch"],
}


class FakeStream:
    def __init__(self, text, stop="end_turn"):
        self.msg = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop, stop_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=100))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, refuse=False):
        self.calls = []
        self.refuse = refuse
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **kw):
        self.calls.append(kw)
        if self.refuse:
            return FakeStream("", "refusal")
        prompt = kw["messages"][0]["content"][-1]["text"]
        if "format" in kw["output_config"]:
            return FakeStream(json.dumps(PLAN))
        if "specification for" in prompt.lower() and "write the complete" not in prompt.lower():
            return FakeStream("spec")
        name = "model" if "`model.py`" in prompt.split("Write the complete contents of")[-1] else "train"
        return FakeStream(f"Here:\n```python\n# {name}\n```\n")


def test_pipeline_writes_repo(tmp_path):
    client = FakeClient()
    paper = {"type": "text", "text": "paper", "cache_control": {"type": "ephemeral"}}
    PaperCoder(paper, client=client, log=lambda *a: None).run(tmp_path)
    assert (tmp_path / "model.py").read_text() == "# model\n"
    assert (tmp_path / "train.py").read_text() == "# train\n"
    assert (tmp_path / "config.yaml").read_text() == "lr: 0.001\n"
    assert "batch size" in (tmp_path / "README.md").read_text()
    # 1 plan + 2 analyses + 2 code calls, every one with the paper as the cached first block
    assert len(client.calls) == 5
    assert all(c["messages"][0]["content"][0] is paper for c in client.calls)
    assert all(c["fallbacks"] == "default" and c["model"] == "claude-opus-5-5" for c in client.calls)
    # train.py is written after model.py and sees it
    assert "# model" in client.calls[-1]["messages"][0]["content"][-1]["text"]


def test_refusal_raises(tmp_path):
    with pytest.raises(RefusedError):
        PaperCoder({"type": "text", "text": "x"}, client=FakeClient(refuse=True), log=lambda *a: None).run(tmp_path)


def test_strip_fences():
    assert strip_fences("blah\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert strip_fences("x = 2") == "x = 2\n"
