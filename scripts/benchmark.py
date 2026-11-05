"""Measure the local model on this machine and write BENCHMARKS.md.

    python scripts/benchmark.py [--model path] [--runs 3]

Reports prompt-processing and generation tokens per second, wall time, and
peak resident memory, together with what the machine is — so every number
in the README can be traced to a run. Numbers vary with quantization,
threads, context and thermals; that is why this script exists.
"""

from __future__ import annotations

import argparse
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from edge_node import netguard
from edge_node.llm import LocalLLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=128)
args = ap.parse_args()

PROMPT = (
    "In three sentences, explain why a self-contained machine that senses and acts "
    "on its surroundings without any network connection is still useful today."
)


def cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:  # pragma: no cover
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


llm = LocalLLM(args.model)
gen_tps: list[float] = []
walls: list[float] = []
with netguard():
    llm.chat([{"role": "user", "content": "warm up"}], max_tokens=8)   # load weights, prime caches
    for _ in range(args.runs):
        r = llm.chat([{"role": "user", "content": PROMPT}], max_tokens=args.max_tokens)
        gen_tps.append(r.tokens_per_second)
        walls.append(r.seconds)

_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _divisor
report = f"""# Benchmarks

Measured by `scripts/benchmark.py`, offline (netguard active), on:

- machine: {cpu_name()} — {platform.system()} {platform.release()} ({platform.machine()})
- python: {platform.python_version()}
- model: `{Path(args.model).name}` (llama.cpp, default threads)
- prompt: {len(PROMPT)} chars; {args.max_tokens} max new tokens; {args.runs} runs after a warm-up
- date: {time.strftime('%Y-%m-%d')}

| metric | value |
|---|---|
| generation, tokens/s (median) | {statistics.median(gen_tps):.1f} |
| generation, tokens/s (min–max) | {min(gen_tps):.1f}–{max(gen_tps):.1f} |
| wall time per reply (median) | {statistics.median(walls):.2f} s |
| peak resident memory | {peak_mb:.0f} MB |

These are the numbers for *this* machine and *this* quantization. Hardware-tier
figures in the README are sourced from vendors and community benchmarks unless
marked as measured here.
"""
Path("BENCHMARKS.md").write_text(report)
print(report)
