"""Offline speech in and out (optional extra, not covered by the test suite).

Thin wrappers over two well-established local tools, called as
subprocesses so nothing here phones home: ``whisper-cli`` from whisper.cpp
for speech-to-text and ``piper`` for text-to-speech. Both need a one-time
model download (a network step, documented in scripts/download_models.sh).
Round trip with small models is several seconds — not "real time".
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def transcribe(wav_path: str | Path, model_path: str | Path = "models/ggml-base.en.bin") -> str:
    exe = shutil.which("whisper-cli") or shutil.which("whisper-cpp") or shutil.which("main")
    if not exe:
        raise FileNotFoundError("whisper.cpp binary not found on PATH (install whisper-cpp)")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"no whisper model at {model_path}")
    out = subprocess.run(
        [exe, "-m", str(model_path), "-f", str(wav_path), "-nt", "-np"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def speak(text: str, out_wav: str | Path, voice_path: str | Path = "models/en_GB-alan-medium.onnx") -> Path:
    exe = shutil.which("piper")
    if not exe:
        raise FileNotFoundError("piper binary not found on PATH (pip install piper-tts)")
    if not Path(voice_path).exists():
        raise FileNotFoundError(f"no piper voice at {voice_path}")
    subprocess.run(
        [exe, "--model", str(voice_path), "--output_file", str(out_wav)],
        input=text, text=True, check=True, capture_output=True,
    )
    return Path(out_wav)
