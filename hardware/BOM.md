# Bill of materials — three honest tiers

Every figure below is either **measured** (by `scripts/benchmark.py` on hardware I own, recorded in `BENCHMARKS.md`) or **sourced** (vendor specifications and community benchmarks, cited in the README). Prices are approximate launch or street prices in late 2026 and move with stock and tariffs.

## Tier 0 — a laptop, £0

Any recent x86 or Arm laptop, CPU only. Runs `demo_chat.py` and `demo_rag.py` as they are. Its purpose is to show the software works anywhere before you buy anything. A 0.5B-parameter model at Q4 runs comfortably on 8 GB of RAM.

## Tier 1 — perception node, about £200–£260

| Part | ~£ | Note |
|---|---|---|
| Raspberry Pi 5, 8 GB | 75 | the host |
| Raspberry Pi AI HAT+ with Hailo-8 (26 TOPS) | 90 | PCIe Gen 3; the 13-TOPS Hailo-8L variant is ~£55 |
| Camera Module 3 | 25 | — |
| USB microphone | 10 | for the voice loop |
| Active cooler | 5 | required: without it throughput drops roughly 30 % within half an hour |
| NVMe HAT + 256 GB SSD, or a fast SD card | 30 / 10 | models and documents |
| Power supply (27 W USB-C) | 12 | — |

Sourced strengths: object detection in the tens of frames per second (YOLOv8s around 60–80 FPS on the Hailo-8, against roughly 5–8 FPS on the Pi's CPU alone); a 1B-class language model at roughly 6–8 tokens per second on the CPU; a fully offline voice round-trip of several seconds with small models.

## Tier 2 — reasoning node, about £300

| Part | ~£ | Note |
|---|---|---|
| NVIDIA Jetson Orin Nano Super Developer Kit | 250 | launched at $249; street prices have run higher |
| NVMe SSD | 30 | — |
| Camera (CSI or USB) | 25 | — |

Sourced strengths: a GPU-accelerated language model — 1B–4B models comfortable; NVIDIA reports roughly 28 tokens per second for Llama 3.1 8B at W4A16 through vLLM, though 7–8B models are tight in 8 GB. The headline "67 TOPS" is a sparse-INT8 peak, not a tokens-per-second figure; memory capacity and bandwidth (8 GB, 102 GB/s) are the real limit.

## Optional add-on — Google Coral USB Accelerator, about £60–£75

An int8 CNN coprocessor (4 TOPS) for object detection only. It **does not run language models**: it has about 8 MB of on-chip memory and no floating-point kernels, so transformer workloads fall back to the CPU. Google's separate "Coral NPU" is a different chip.
