"""Utilities for loading cached PyMove reference results.

Populated by running the real ``pymove.utils.integration.join_with_pois``/
``join_with_pois_by_category``/``join_with_events`` once inside a dedicated
``.venv-pymove`` (see ``tests/populate_pymove_cache.py`` /
``scripts/populate_pymove_cache.sh``), against the exact fixtures baked
into PyMove's own docstring examples.

The resulting parquet files are committed here; fastmob's normal test
suite never invokes pymove itself at test time, only the committed cache
-- unlike TransBigData, PyMove needs Python <=3.10 and a pandas<1.4 pin
incompatible with fastmob's normal .venv (see CLAUDE.md and this repo's
``dev-pymove`` comment in pyproject.toml).

Cache schema (``tests/shared/pymove_reference/<dataset>/``):
    doc_example_poi/
        input.parquet, pois.parquet
        join_with_pois.parquet, join_with_pois_by_category.parquet
    doc_example_events/
        input.parquet, events.parquet
        join_with_events.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REFERENCE_DIR = Path(__file__).parent / "pymove_reference"


class PymoveReferenceDataset:
    """Cached PyMove POI/event join output for one fixture."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    def _read(self, filename: str) -> pd.DataFrame | None:
        path = self._dir / filename
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def input_traj(self) -> pd.DataFrame | None:
        return self._read("input.parquet")

    def pois(self) -> pd.DataFrame | None:
        return self._read("pois.parquet")

    def events(self) -> pd.DataFrame | None:
        return self._read("events.parquet")

    def join_with_pois(self) -> pd.DataFrame | None:
        return self._read("join_with_pois.parquet")

    def join_with_pois_by_category(self) -> pd.DataFrame | None:
        return self._read("join_with_pois_by_category.parquet")

    def join_with_events(self) -> pd.DataFrame | None:
        return self._read("join_with_events.parquet")
