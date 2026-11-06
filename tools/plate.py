"""Draw the node as a plate: six layers inside the air gap, the inputs, the
one setup-time crossing, and a scenario table of the three hardware tiers
with their measured or sourced figures.  Run:  python tools/plate.py

Paper, ink, one accent. Every number on the plate is the README's; the
figure marked measured comes from BENCHMARKS.md, the rest are sourced.
"""

from __future__ import annotations

from pathlib import Path

INK = "#111111"
INK2 = "#6b6b6b"
RULE = "#d4d4d4"
PAPER = "#ffffff"
PAPER2 = "#f4f4f4"
ACCENT = "#0000ee"
FONT = "IBM Plex Mono, ui-monospace, Menlo, monospace"

LAYERS: list[tuple[str, str, str]] = [
    ("inputs", "camera · microphone · a documents folder · keyboard", ""),
    ("perception", "YOLOv8 detection — Hailo-8, or ONNX on the CPU", "optional"),
    ("speech", "whisper.cpp in · Piper out", "optional"),
    ("knowledge", "chunk → embed → cosine retrieve → cite", "rag.py"),
    ("reasoning", "a local model through llama.cpp, GGUF, temperature 0", "llm.py"),
    ("orchestrator", "input → optional perception → retrieve → generate → output", "orchestrator.py"),
]

TIERS: list[tuple[str, str, str]] = [
    ("tier 0", "any recent laptop, CPU only", "£0 — measured: 48.8 tokens/s, 635 MB peak, 0.5B Q4 on an Apple M1"),
    ("tier 1", "Raspberry Pi 5 8 GB + AI HAT+ (Hailo-8) + camera + mic", "about £200–260 — sourced: 6–8 tokens/s for a 1B model; 60–80 FPS YOLOv8s"),
    ("tier 2", "Jetson Orin Nano Super developer kit + NVMe + camera", "about £300 — sourced: 1B–4B models on the GPU; 8 GB is the real limit"),
]


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;")


def _text(x: float, y: float, s: str, size: float = 10, fill: str = INK, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{_esc(s)}</text>'


def plate() -> str:
    """The plate as one SVG string."""
    pad, w = 48, 864
    box_x, box_w = pad + 20, 520
    layer_h, layer_gap = 44, 12
    top = pad + 42
    gap_h = 28 + len(LAYERS) * (layer_h + layer_gap) + 16
    body = _text(pad, pad - 14, "offline-edge-node / six layers inside an air gap", 10, INK2)
    body += _text(pad + w, pad - 14, "benmicallef / offline-edge-node", 10, INK2, "end")
    # the air gap: a dashed boundary nothing crosses at run time
    body += (
        f'<rect x="{box_x - 14}" y="{top - 8}" width="{box_w + 28}" height="{gap_h}" fill="none" '
        f'stroke="{INK}" stroke-dasharray="6 5" stroke-width="1.2"/>'
    )
    body += _text(box_x, top + 8, "NO NETWORK — enforced by edge_node.netguard: any outbound connection or DNS lookup raises", 8, INK2)
    y = top + 28
    for i, (name, what, note) in enumerate(LAYERS):
        body += f'<rect x="{box_x}" y="{y}" width="{box_w}" height="{layer_h}" fill="{PAPER2 if i else PAPER}" stroke="{INK}" stroke-width="0.8"/>'
        body += _text(box_x + 10, y + 18, name, 10, INK)
        body += _text(box_x + 10, y + 33, what, 8.5, INK2)
        if note:
            body += _text(box_x + box_w - 10, y + 18, note, 8, ACCENT if note == "optional" else INK2, "end")
        if i:
            body += f'<path d="M{box_x + 60} {y - layer_gap}v{layer_gap - 4}m-3 -4l3 4l3 -4" fill="none" stroke="{INK}" stroke-width="0.8"/>'
        y += layer_h + layer_gap
    # the one crossing: a setup-time download, drawn in the accent, dashed
    cx = box_x + box_w / 2
    body += f'<path d="M{cx} {top + gap_h + 30}V{top + gap_h - 8}m-3 6l3 -6l3 6" fill="none" stroke="{ACCENT}" stroke-dasharray="3 3" stroke-width="1"/>'
    body += _text(cx + 8, top + gap_h + 26, "one-time model download (scripts/download_models.sh) — setup only, never at run time", 8, ACCENT)
    # the loop it closes, at the right
    rx = box_x + box_w + 60
    body += _text(rx, top + 8, "the loop it closes", 9, INK2)
    loop = [
        "a question, or a frame, or speech",
        "retrieval confines the model to your documents",
        "the model answers, citing what it retrieved",
        "the answer is read, spoken, or acted on by you",
        "nothing has left the box",
    ]
    for i, line in enumerate(loop):
        ly = top + 36 + i * 30
        body += f'<circle cx="{rx + 4}" cy="{ly - 3}" r="3" fill="none" stroke="{INK}" stroke-width="0.8"/>'
        body += _text(rx + 14, ly, line, 8.5, INK)
        if i < len(loop) - 1:
            body += f'<line x1="{rx + 4}" y1="{ly + 1}" x2="{rx + 4}" y2="{ly + 23}" stroke="{RULE}"/>'
    body += _text(rx, top + 36 + len(loop) * 30 + 6, "what it will not do: drive your screen unsupervised.", 8, INK2)
    body += _text(rx, top + 36 + len(loop) * 30 + 18, "a 1–4B model on 8 GB cannot do that reliably, and", 8, INK2)
    body += _text(rx, top + 36 + len(loop) * 30 + 30, "letting it try unconfirmed is a risk, not a feature.", 8, INK2)
    # scenario table: the three tiers
    ty = top + gap_h + 60
    body += f'<line x1="{pad}" y1="{ty - 14}" x2="{pad + w}" y2="{ty - 14}" stroke="{RULE}"/>'
    rows: list[tuple[str, str]] = [
        ("project", "offline-edge-node — local chat and retrieval over your own documents, with offline enforced"),
        ("programme", "six layers in one box: inputs, perception, speech, knowledge, reasoning, orchestrator; the dashed line is the point"),
    ]
    for tier, parts, cost in TIERS:
        rows.append((tier, f"{parts} / {cost}"))
    rows.append(("record", "18 tests; the air gap is proven by a test against raw sockets, DNS and urllib; the retrieval test runs inside it"))
    yy = ty
    for key, value in rows:
        lines: list[str] = []
        cur = ""
        for word in value.split(" "):
            if cur and len(cur) + 1 + len(word) > 112:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        if cur:
            lines.append(cur)
        body += _text(pad, yy, key, 10, INK2)
        for i, line in enumerate(lines):
            body += _text(pad + 100, yy + i * 13, line, 10, INK)
        yy += len(lines) * 13 + 5
    body += _text(pad, yy + 14, "drawn by tools/plate.py — paper, ink, one accent. prices rough, late 2026; measured means scripts/benchmark.py on hardware I own.", 8, INK2)
    height = yy + 14 + pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w + 2 * pad} {height}" width="{w + 2 * pad}" height="{height}" '
        'role="img" aria-label="offline-edge-node: six layers inside an air gap, and three hardware tiers">\n'
        f'<rect width="{w + 2 * pad}" height="{height}" fill="{PAPER}"/>\n'
        f'<rect x="0.5" y="0.5" width="{w + 2 * pad - 1}" height="{height - 1}" fill="none" stroke="{RULE}"/>\n'
        f"{body}\n</svg>\n"
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "plate.svg"
    out.write_text(plate())
    print(f"wrote {out.relative_to(out.parent.parent)}")
