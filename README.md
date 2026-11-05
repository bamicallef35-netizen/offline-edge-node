# offline-edge-node

A fully offline, air-gapped perception + reasoning + knowledge node. Local chat, retrieval over your own documents with citations, and optional on-device vision and voice — on a laptop today, on a £200 Raspberry Pi or a £300 Jetson tomorrow — with *offline* enforced by a guard that makes any network attempt crash, where most projects only assert it.

This is the honest, buildable version of a tempting idea: a small box that sees, listens, reads what you give it, and answers, with nothing leaving the device. What it is not is an autonomous agent that drives your computer; the section below says why.

```bash
pip install -e ".[dev]"            # core needs numpy only
pytest -q                          # the air-gap guard + offline retrieval, no model needed
python demos/demo_rag.py "What was Musicolour?"     # retrieval, offline, right now

pip install -e ".[llm]"            # llama.cpp (compiles; a few minutes)
bash scripts/download_models.sh    # the ONE network step: a ~400 MB Apache-2.0 model
python demos/demo_chat.py          # chat, air-gapped, with measured tokens/s
python scripts/benchmark.py        # writes BENCHMARKS.md for this machine
```

## What this does (offline, today)

- **Local chat.** A small instruction-tuned model (Qwen2.5-Instruct, Q4 GGUF, Apache-2.0) through llama.cpp, temperature 0, with tokens-per-second printed after every reply. Measured on an Apple M1 laptop with 8 GB: the 0.5B Q4 model generates at a median 48.8 tokens/s with 635 MB peak resident memory, and a grounded RAG answer runs at about 15 tokens/s because its prompt is longer — see [`BENCHMARKS.md`](BENCHMARKS.md), written by the repo's own benchmark. A 1B-class model runs at roughly 6–8 tokens/s on a Raspberry Pi 5 CPU (sourced, not measured by me) and faster on a Jetson GPU.
- **Retrieval over your documents.** `docs/` is chunked, embedded and searched by cosine similarity; answers are grounded in the retrieved passages and cite them as `[file#chunk]`. The default embedder is a dependency-free lexical baseline so this works with no model at all; a local GGUF embedding model slots in for semantic retrieval.
- **On-device vision (optional).** YOLOv8 object detection through ONNX Runtime on the CPU, or one to two orders of magnitude faster on a Raspberry Pi 5 with a Hailo-8 AI HAT+ (around 60–80 FPS for YOLOv8s, sourced). Detection means boxes and labels — not scene understanding.
- **Offline voice (optional).** whisper.cpp in, Piper out, as local subprocesses. A round trip is several seconds with small models; it is not real-time.
- **An enforced air gap.** `edge_node.netguard` patches the process's socket entry points so any outbound connection or DNS lookup raises. [`tests/test_netguard.py`](tests/test_netguard.py) proves it against raw sockets, `create_connection`, DNS and `urllib`; the retrieval test runs inside it. If a dependency ever tries to phone home, the demo dies instead of leaking.

## What this does NOT do

It is not an autonomous computer-use agent, and the README will not pretend it could be. Reliable navigation of an arbitrary GUI is unsolved even for the best cloud models: Claude Sonnet 4.5 scored 61.4 % on the OSWorld benchmark against a 72.4 % human baseline (September 2025), running with vastly more compute than any edge device, and a 1–4B model on 8 GB cannot do it dependably. Small vision-language models are also error-prone at reading screens. If this node ever acts on a computer, it will be one human-confirmed action at a time on a known application — a future direction, stated as such.

Likewise: TOPS figures are peak sparse-INT8 marketing numbers, not model speed; 7–8B models are memory-tight on an 8 GB board; the Coral Edge TPU accelerates CNN detection only and cannot run language models; and every throughput number depends on quantization, threads, context and thermals — which is why the benchmark script exists and why unlabelled numbers are not quoted here.

## Legality and privacy

Local inference on your own hardware is lawful. The only compliance surface is each model's licence; the defaults here are Apache-2.0 (Qwen2.5, nomic-embed-text, Moondream2) so a public portfolio carries no additional use conditions — Llama and Gemma are fine alternatives but carry their own terms. Keeping every document, image and utterance on the device is the privacy point of the design, not a side effect.

## Hardware

[`hardware/BOM.md`](hardware/BOM.md) gives three tiers — a £0 laptop, a ~£230 Raspberry Pi 5 perception node, a ~£300 Jetson Orin Nano reasoning node — with each number marked measured or sourced, and [`hardware/architecture.md`](hardware/architecture.md) draws the six layers inside the air gap.

## Why this sits with the rest

A self-contained machine that builds its picture of the world from its own sensors, with no signal from outside, is the modern form of Grey Walter's tortoise and the local-first cousin of navigation without GPS. The engineering concerns are the ones that matter in autonomy: inference on a budget, quantization, latency and memory, accelerator integration, and a sensor-to-decision loop you can trust because you can test what it will not do.

## License

MIT — see [`LICENSE`](LICENSE).
