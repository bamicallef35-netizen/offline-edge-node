"""Speak a question, hear an answer — all local (optional extra).

    python demos/demo_voice.py question.wav

Needs whisper.cpp (`whisper-cli`) and Piper on PATH plus their models in
models/ (a one-time network step), and the chat model. Round trip is several
seconds with small models; that is stated, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

from edge_node import netguard

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)

try:
    from edge_node.llm import LocalLLM
    from edge_node.voice import speak, transcribe
except ImportError as e:
    print(e)
    sys.exit(0)

with netguard():
    try:
        question = transcribe(sys.argv[1])
        print("heard:", question)
        reply = LocalLLM(str(MODEL)).chat([{"role": "user", "content": question}], max_tokens=120)
        print("answer:", reply.text)
        out = speak(reply.text, ROOT / "answer.wav")
        print("spoken to", out)
    except FileNotFoundError as e:
        print(e)
