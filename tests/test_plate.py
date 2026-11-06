"""The plate is one well-formed SVG carrying the README's figures."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from plate import plate  # noqa: E402


def test_plate_is_one_svg_with_the_layers_and_tiers() -> None:
    svg = plate()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    words = (
        "inputs", "perception", "speech", "knowledge", "reasoning", "orchestrator",
        "netguard", "tier 0", "tier 2", "48.8",
    )
    for word in words:
        assert word in svg
    assert plate() == svg
    assert not re.search(r"\d\.\d+\.\d", svg)
