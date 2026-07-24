"""Utilities for loading cached TransBigData reference results.

Populated by running the real ``transbigdata.traj_mapmatch`` once (see
``tests/populate_transbigdata_cache.py`` / ``scripts/populate_transbigdata_cache.sh``)
against a small synthetic road network built as an osmnx-style
``networkx.MultiDiGraph`` (nodes carry ``x``/``y`` lon/lat attributes, edges
carry a ``geometry`` LineString attribute) -- the exact input shape
``traj_mapmatch`` requires ("a networkx multidigraph, created by osmnx").
``osmnx``/``networkx`` are only ever used inside this comparison harness,
never by fastmob's own code (see CLAUDE.md).

The resulting parquet files are committed here; fastmob's normal test suite
never invokes transbigdata/osmnx/networkx at test time. TransBigData's own
pinned stack is individually compatible with fastmob's normal ``.venv``
(``pandas>=3``, ``numpy>=2``, ``geopandas>=1.1``, ``shapely>=2.1``), but
installing it there anyway triggers a dedicated ``.venv-transbigdata``:
installing transbigdata's extras (transitively: pykalman, scikit-base) into
an *already-populated* .venv can trigger a full dependency re-resolution
that silently downgrades unrelated packages (numpy in particular) --
see pyproject.toml's ``dev-transbigdata`` comment.

Cache schema (``tests/shared/transbigdata_reference/<dataset>/``):
    nodes.parquet:   node_idx, lat, lng          (fastmob RoadNetwork input)
    edges.parquet:   from_node, to_node, length_m, weight_ds
    input.parquet:   lat, lng                    (trajectory points to match)
    matched.parquet: lat, lng, dist_m             (TransBigData's snapped result)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REFERENCE_DIR = Path(__file__).parent / "transbigdata_reference"


class TransBigDataReferenceDataset:
    """Cached TransBigData ``traj_mapmatch`` output for one synthetic dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    def _read(self, filename: str) -> pd.DataFrame | None:
        path = self._dir / filename
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def nodes(self) -> pd.DataFrame | None:
        return self._read("nodes.parquet")

    def edges(self) -> pd.DataFrame | None:
        return self._read("edges.parquet")

    def input_traj(self) -> pd.DataFrame | None:
        return self._read("input.parquet")

    def matched(self) -> pd.DataFrame | None:
        return self._read("matched.parquet")

    def meta(self) -> dict | None:
        path = self._dir / "meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
