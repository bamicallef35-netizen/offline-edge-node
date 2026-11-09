"""Interactive local chat, air-gapped, with measured tokens per second.

    python demos/demo_chat.py [--model models/qwen2.5-0.5b-instruct-q4_k_m.gguf]
"""

from __future__ import annotations

import argparse

from edge_node import netguard
from edge_node.llm import LocalLLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
ap.add_argument("--max-tokens", type=int, default=200)
args = ap.parse_args()

llm = LocalLLM(args.model)
history: list[dict[str, str]] = [
    {"role": "system", "content": "You are a concise assistant running offline on this machine."}
]
print(f"model: {args.model}\nnetwork: blocked (netguard). Ctrl-D to quit.\n")
with netguard():
    while True:
        try:
            user = input("you > ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = llm.chat(history, max_tokens=args.max_tokens)
        history.append({"role": "assistant", "content": reply.text})
        print(f"\n{reply.text}\n   [{reply.completion_tokens} tokens, {reply.tokens_per_second:.1f} tok/s]\n")
