"""edge_node — a fully offline perception + reasoning + knowledge node.

Local chat, retrieval over your own documents, and (optionally) on-device
vision and voice, with no network — enforced by ``netguard``, where most
projects only assert it. README.md lists what it can and cannot do.
"""

from .netguard import NetworkAttempt, netguard
from .rag import Chunk, HashedEmbedder, Index, chunk_text, grounded_prompt, load_docs

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "HashedEmbedder",
    "Index",
    "NetworkAttempt",
    "chunk_text",
    "grounded_prompt",
    "load_docs",
    "netguard",
]
