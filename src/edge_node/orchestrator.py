"""Route a question through retrieval and the local model, inside the guard."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LocalLLM, Reply
from .netguard import netguard
from .rag import Chunk, Index, grounded_prompt

SYSTEM = (
    "You are a careful assistant running entirely offline on this device. "
    "Be brief and cite your sources by their bracketed references."
)


@dataclass(frozen=True)
class Answer:
    reply: Reply
    citations: list[Chunk]
    prompt: str


def answer(question: str, index: Index, llm: LocalLLM, *, k: int = 3) -> Answer:
    """Retrieve, ground, generate — with the network blocked throughout."""
    with netguard():
        hits = index.query(question, k=k)
        prompt = grounded_prompt(question, hits)
        reply = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        )
    return Answer(reply, [c for _, c in hits], prompt)
