"""The oldest Narwhals that CI tests must be the declared lower bound."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ci_pins_the_declared_narwhals_floor():
    declared = re.search(r'"narwhals>=([\d.]+)"', (ROOT / "pyproject.toml").read_text())
    pinned = re.search(r"narwhals==([\d.]+)", (ROOT / ".github/workflows/CI.yml").read_text())
    assert declared and pinned, "narwhals bound or CI pin not found"
    assert pinned.group(1) == f"{declared.group(1)}.0"
