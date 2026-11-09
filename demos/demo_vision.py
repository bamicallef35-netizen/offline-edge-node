"""Detect objects in an image, on the CPU, offline (optional extra).

    pip install 'offline-edge-node[vision]'
    python demos/demo_vision.py path/to/image.jpg [--model models/yolov8n.onnx]

Needs a YOLOv8n ONNX export in models/ (a one-time network step). Prints
boxes and labels; that is the whole claim.
"""

from __future__ import annotations

import argparse
import sys

from edge_node import netguard

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", default="models/yolov8n.onnx")
ap.add_argument("--conf", type=float, default=0.4)
args = ap.parse_args()

try:
    import onnxruntime  # noqa: F401

    from edge_node.vision import detect
except ImportError:
    print("onnxruntime/pillow not installed: pip install 'offline-edge-node[vision]'")
    sys.exit(0)

with netguard():
    try:
        dets = detect(args.image, args.model, conf=args.conf)
    except FileNotFoundError as e:
        print(e)
        sys.exit(0)
for d in dets:
    x1, y1, x2, y2 = (round(v) for v in d.box)
    print(f"{d.label:14s} {d.confidence:.2f}  [{x1}, {y1}, {x2}, {y2}]")
print(f"{len(dets)} detections, offline")
