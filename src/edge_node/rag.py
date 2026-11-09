"""Retrieval over a folder of local documents, with citations.

Chunk the documents deterministically, embed each chunk, retrieve the
nearest chunks for a question by cosine similarity, and assemble a prompt
that cites them. Nothing leaves the machine.

Two embedders are provided. ``HashedEmbedder`` is a dependency-free lexical
baseline (feature-hashed term frequencies): deterministic, fast, and good
enough for exact-ish retrieval and for the test suite, but it is not a
semantic model. ``LlamaEmbedder`` wraps a local GGUF embedding model through
llama.cpp for real semantic retrieval; it is optional.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "its", "was", "for",
    "as", "by", "at", "with", "that", "this", "from", "be", "are", "were", "he", "his",
}


@dataclass(frozen=True)
class Chunk:
    doc: str
    index: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc}#{self.index}"


def chunk_text(text: str, *, size: int = 700, overlap: int = 1) -> list[str]:
    """Split on blank lines, then pack paragraphs into windows of at most
    ``size`` characters, carrying ``overlap`` trailing paragraphs into the
    next window so a thought that straddles a boundary is not lost."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    window: list[str] = []
    length = 0
    for p in paras:
        if window and length + len(p) + 2 > size:
            chunks.append("\n\n".join(window))
            window = window[-overlap:] if overlap else []
            length = sum(len(w) + 2 for w in window)
        window.append(p)
        length += len(p) + 2
    if window:
        chunks.append("\n\n".join(window))
    return chunks


def load_docs(folder: str | Path, patterns: Sequence[str] = ("*.md", "*.txt")) -> dict[str, str]:
    """Read every matching file in ``folder`` as {filename: text}, sorted."""
    root = Path(folder)
    docs: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            docs[path.name] = path.read_text(encoding="utf-8")
    return docs


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class HashedEmbedder:
    """Feature-hashed term-frequency vectors, L2-normalised. Deterministic,
    no model, no network. A lexical baseline — not semantic."""

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _tokens(text):
                h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
                bucket = int.from_bytes(h, "little") % self.dim
                sign = 1.0 if h[0] & 1 else -1.0
                out[row, bucket] += sign * (1.0 + math.log(1 + text.lower().count(tok)))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class LlamaEmbedder:
    """Semantic embeddings from a local GGUF model via llama.cpp (optional)."""

    def __init__(self, model_path: str, n_threads: int | None = None) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'offline-edge-node[llm]' for LlamaEmbedder") from e
        self._llm = Llama(model_path=model_path, embedding=True, n_threads=n_threads, verbose=False)
        self.dim = int(self._llm.n_embd())

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vecs = np.asarray(self._llm.embed(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(vecs / norms, dtype=np.float32)


class Index:
    """A tiny in-memory vector index: brute-force cosine over numpy.

    Fine for a few thousand chunks, which is what a personal document folder
    is. sqlite-vec is the documented step up if you outgrow it.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: NDArray[np.float32] = np.zeros((0, embedder.dim), dtype=np.float32)

    def build(self, docs: dict[str, str], **chunk_kwargs: int) -> Index:
        self.chunks = [
            Chunk(name, i, text)
            for name in sorted(docs)
            for i, text in enumerate(chunk_text(docs[name], **chunk_kwargs))
        ]
        self.vectors = self.embedder.embed([c.text for c in self.chunks])
        return self

    def query(self, question: str, k: int = 3) -> list[tuple[float, Chunk]]:
        if not self.chunks:
            return []
        q = self.embedder.embed([question])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]


def grounded_prompt(question: str, hits: Sequence[tuple[float, Chunk]]) -> str:
    """Assemble a prompt that confines the model to the retrieved passages
    and asks it to cite them by reference."""
    context = "\n\n".join(f"[{c.ref}]\n{c.text}" for _, c in hits)
    return (
        "Answer the question using only the passages below. Cite the passage "
        "reference in square brackets after each claim. If the passages do not "
        "contain the answer, say so.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
