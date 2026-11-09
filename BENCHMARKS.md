# Benchmarks

Measured by `scripts/benchmark.py`, offline (netguard active), on:

- machine: Apple M1 — Darwin 23.2.0 (arm64)
- python: 3.13.5
- model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (llama.cpp, default threads)
- prompt: 151 chars; 128 max new tokens; 3 runs after a warm-up
- date: 2026-11-05

| metric | value |
|---|---|
| generation, tokens/s (median) | 48.8 |
| generation, tokens/s (min–max) | 42.8–49.1 |
| wall time per reply (median) | 2.01 s |
| peak resident memory | 635 MB |

These are the numbers for *this* machine and *this* quantization. Hardware-tier
figures in the README are sourced from vendors and community benchmarks unless
marked as measured here.
