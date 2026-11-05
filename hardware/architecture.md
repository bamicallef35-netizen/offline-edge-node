# Architecture

Six layers in one air-gapped box. The dashed boundary is the point: nothing crosses it at run time.

```
┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   NO NETWORK  (enforced by edge_node.netguard)

   INPUTS        camera · microphone · docs/ folder · keyboard
        │
   PERCEPTION    YOLOv8 detection (Hailo-8, or ONNX on CPU)   [optional]
   SPEECH        whisper.cpp in · Piper out                    [optional]
        │
   KNOWLEDGE     chunk → embed → cosine retrieve → cite        (rag.py)
        │
   REASONING     local LLM via llama.cpp, GGUF, temperature 0  (llm.py)
        │
   ORCHESTRATOR  input → optional perception → retrieve → generate → output
                                                               (orchestrator.py)
└ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
          ▲
          ┆  one-time model download (scripts/download_models.sh) — setup only
```

**Out of scope, deliberately:** autonomous computer-use. Reliable GUI navigation is unsolved even for frontier cloud models — Claude Sonnet 4.5 scored 61.4 % on OSWorld against a 72.4 % human baseline in September 2025 — and those run with far more compute than any edge device. A 1–4B model on 8 GB cannot do it reliably, and letting it try without a human confirming each action is a risk, not a feature. If this node ever drives a screen, it will be one confirmed action at a time, on a known application.

**Why these pieces:** the local model is the reasoning layer, retrieval keeps it honest by confining it to your own documents, perception gives it eyes, speech gives it a mouth, and the guard makes the word "offline" mean something.
