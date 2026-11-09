#!/usr/bin/env bash
# The ONE step that touches the network. Everything else runs air-gapped.
# Default models are Apache-2.0. Re-run safely; existing files are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

fetch() {  # fetch <url> <file>
  if [ -f "models/$2" ]; then echo "have models/$2"; return; fi
  echo "downloading $2"
  curl -L --fail --retry 3 --progress-bar -o "models/$2" "$1"
}

# Reasoning: Qwen2.5-Instruct, Q4_K_M. 0.5B runs on anything; 1.5B is the
# comfortable default on a laptop or a Jetson Orin Nano.
fetch "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
      "qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [ "${EDGE_NODE_LARGE:-0}" = "1" ]; then
  fetch "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi

# Semantic embeddings (optional; the hashed embedder needs no model):
if [ "${EDGE_NODE_EMBED:-0}" = "1" ]; then
  fetch "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf" \
        "nomic-embed-text-v1.5.Q4_K_M.gguf"
fi

echo "done. From here on, unplug."
