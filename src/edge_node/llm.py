"""A local language model through llama.cpp, with measured throughput.

Optional: install with ``pip install 'offline-edge-node[llm]'``. Default
models are Apache-2.0 (Qwen2.5-Instruct GGUF); see scripts/download_models.sh,
which is the only step that touches the network.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Reply:
    text: str
    completion_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        seed: int = 0,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "llama-cpp-python is not installed; run: pip install 'offline-edge-node[llm]'"
            ) from e
        self.model_path = model_path
        self._llm = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, seed=seed, verbose=False
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Reply:
        """One deterministic (temperature 0) chat completion, timed."""
        t0 = time.perf_counter()
        out: Any = self._llm.create_chat_completion(
            messages=list(messages), max_tokens=max_tokens, temperature=temperature
        )
        seconds = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"] or ""
        tokens = int(out.get("usage", {}).get("completion_tokens", 0))
        return Reply(text.strip(), tokens, seconds)

    def stream(self, messages: Sequence[dict[str, str]], *, max_tokens: int = 256) -> Iterator[str]:
        """Yield text deltas as they are generated."""
        for part in self._llm.create_chat_completion(
            messages=list(messages), max_tokens=max_tokens, temperature=0.0, stream=True
        ):
            delta = part["choices"][0]["delta"].get("content")
            if delta:
                yield delta
