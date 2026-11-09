"""Loads the default model if it is present; skipped otherwise, so CI never
downloads a model and the suite reports what it actually exercised."""

from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="no local model; run scripts/download_models.sh")


def test_model_answers_deterministically_offline() -> None:
    pytest.importorskip("llama_cpp")
    from edge_node import netguard
    from edge_node.llm import LocalLLM

    llm = LocalLLM(str(MODEL), n_ctx=512)
    msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
    with netguard():
        a = llm.chat(msgs, max_tokens=8)
        b = llm.chat(msgs, max_tokens=8)
    assert a.text and a.text == b.text
    assert a.completion_tokens > 0 and a.tokens_per_second > 0
