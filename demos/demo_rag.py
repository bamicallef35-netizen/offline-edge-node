"""Ask a question of the documents in docs/, offline, with citations.

    python demos/demo_rag.py "Who built Elmer and Elsie?"

Retrieval always runs. Generation runs if llama-cpp-python and the default
model are present; otherwise the demo prints the retrieved passages, which
is the part of RAG you can check by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

from edge_node import HashedEmbedder, Index, load_docs, netguard

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
question = " ".join(sys.argv[1:]) or "Who built the tortoises Elmer and Elsie, and when?"

with netguard():
    index = Index(HashedEmbedder()).build(load_docs(ROOT / "docs"))
    hits = index.query(question, k=3)

print(f"question: {question}\n")
for score, chunk in hits:
    print(f"[{chunk.ref}]  score {score:.3f}\n{chunk.text[:300].rstrip()}...\n")

try:
    import llama_cpp  # noqa: F401

    from edge_node.llm import LocalLLM
    from edge_node.orchestrator import answer
except ImportError:
    print("(llama-cpp-python not installed: retrieval only. pip install 'offline-edge-node[llm]')")
    sys.exit(0)
if not MODEL.exists():
    print(f"(no model at {MODEL}: retrieval only. run scripts/download_models.sh)")
    sys.exit(0)

result = answer(question, index, LocalLLM(str(MODEL)))
print("answer:\n" + result.reply.text)
print(f"\n[{result.reply.completion_tokens} tokens, {result.reply.tokens_per_second:.1f} tok/s, offline]")
