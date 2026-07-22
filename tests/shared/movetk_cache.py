"""Utilities for loading cached MoveTK reference results.

The cache is populated by hand, once, by building and running the small C++
driver programs in ``../fastmob_benchmarks/benchmarks/simplify/movetk_cpp/``
(see that directory's ``build.sh``) against the same Brightkite slice used
for ``tests/shared/movingpandas_reference/brightkite/input.parquet``,
projected to local planar kilometre coordinates with the same
equirectangular formula as fastmob's own ``project_local_planar_km``. The
resulting JSON files are committed here; fastmob's test suite never invokes
MoveTK or the sibling repo at test time.

Cache schema: ``{"<uid>": [kept row_index, ...], ...}`` -- 0-based row
indices per user that the reference MoveTK algorithm kept, using the same
``row_index`` numbering as
``tests/shared/movingpandas_reference/brightkite/input.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path

_REFERENCE_DIR = Path(__file__).parent / "movetk_reference"


class MovetkReferenceDataset:
    """Cached MoveTK simplification outputs for one dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    def kept_row_index(self, method: str) -> dict[str, set[int]] | None:
        """Cached ``{uid: {kept local row indices}}`` for one method, or None if absent."""
        path = self._dir / f"simplify_{method}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return {uid: set(indices) for uid, indices in raw.items()}
