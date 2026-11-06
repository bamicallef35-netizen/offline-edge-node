"""The pure parts of the detector — box overlap and suppression — need no model."""

from edge_node.vision import Detection, _iou, _nms


def box(x1: float, y1: float, x2: float, y2: float, label: str = "cup", conf: float = 0.9) -> Detection:
    return Detection(label, conf, (x1, y1, x2, y2))


def test_iou_of_identical_boxes_is_one_and_disjoint_is_zero() -> None:
    assert abs(_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) - 1.0) < 1e-9
    assert _iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap() -> None:
    # 10x10 and 10x10 shifted by 5 along x: intersection 50, union 150.
    assert abs(_iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_nms_keeps_the_most_confident_of_overlapping_same_label_boxes() -> None:
    kept = _nms([box(0, 0, 10, 10, conf=0.6), box(1, 1, 11, 11, conf=0.9), box(0, 0, 10, 10, "dog", 0.5)])
    assert [(d.label, d.confidence) for d in kept] == [("cup", 0.9), ("dog", 0.5)]


def test_nms_keeps_non_overlapping_boxes() -> None:
    kept = _nms([box(0, 0, 10, 10), box(50, 50, 60, 60)])
    assert len(kept) == 2
