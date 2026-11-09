"""On-device object detection (optional extra, not covered by the test suite).

Runs a YOLOv8n ONNX model through ONNX Runtime on the CPU. On a Raspberry
Pi 5 the Hailo-8 AI HAT+ runs the same family of models one to two orders
of magnitude faster through Hailo's own runtime; this module is the
portable CPU path. Install with ``pip install 'offline-edge-node[vision]'``
and place ``models/yolov8n.onnx`` (export from ultralytics on another
machine, or download once — a network step).

The scope here is detection: boxes and class names. It is not scene
understanding, it does not read screens, and small models are error-prone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant "
    "stop-sign parking-meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports-ball kite baseball-bat "
    "baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup fork knife spoon bowl "
    "banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch potted-plant "
    "bed dining-table toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink "
    "refrigerator book clock vase scissors teddy-bear hair-drier toothbrush"
).split()


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in input pixels


def _letterbox(img: NDArray[np.uint8], size: int = 640) -> tuple[NDArray[np.float32], float, int, int]:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    from PIL import Image

    resized = np.asarray(Image.fromarray(img).resize((nw, nh)), dtype=np.uint8)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    x = canvas.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), scale, left, top


def detect(
    image_path: str | Path,
    model_path: str | Path = "models/yolov8n.onnx",
    conf: float = 0.4,
) -> list[Detection]:
    try:
        import onnxruntime as ort
        from PIL import Image
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("pip install 'offline-edge-node[vision]' for detection") from e
    if not Path(model_path).exists():
        raise FileNotFoundError(f"no detector at {model_path}; see docstring")
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    x, scale, left, top = _letterbox(img)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # (84, 8400)
    boxes, scores, labels = out[:4].T, out[4:].max(axis=0), out[4:].argmax(axis=0)
    keep = scores >= conf
    dets: list[Detection] = []
    for (cx, cy, w, h), s, c in zip(boxes[keep], scores[keep], labels[keep], strict=True):
        x1, y1 = (cx - w / 2 - left) / scale, (cy - h / 2 - top) / scale
        x2, y2 = (cx + w / 2 - left) / scale, (cy + h / 2 - top) / scale
        dets.append(Detection(COCO[int(c)], float(s), (float(x1), float(y1), float(x2), float(y2))))
    return _nms(dets)


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[Detection], threshold: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.confidence):
        if all(_iou(d, k) < threshold or d.label != k.label for k in kept):
            kept.append(d)
    return kept
