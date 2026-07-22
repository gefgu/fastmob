"""Utilities for loading cached MovingPandas reference results.

The cache is populated once by running:
    bash scripts/populate_movingpandas_cache.sh

in the .venv-movingpandas environment. The resulting files in
tests/shared/movingpandas_reference/ are committed to git so the normal
.venv can run comparison tests without movingpandas installed.

Cache schema (see the project plan's "Cache schema per capability area"):
    input.parquet
        Standardised pandas DataFrame fed to MovingPandas, with an extra
        ``row_index`` column giving each row's 0-based position within its
        user's chronologically-sorted slice (the same local numbering
        fastmob's ``simplify()`` mask-as-indices uses).
    simplify_<method>.parquet
        Columns ``uid`` and ``row_index``: one row per (user, kept local
        row index) pair that MovingPandas' corresponding generalizer kept.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

_REFERENCE_DIR = Path(__file__).parent / "movingpandas_reference"


class MovingPandasReferenceDataset:
    """Cached MovingPandas outputs for one dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    @functools.cached_property
    def input_df(self) -> pd.DataFrame:
        """Standardised pandas DataFrame fed to MovingPandas (uid, datetime, lat, lng, row_index)."""
        return pd.read_parquet(self._dir / "input.parquet")

    def kept_row_index(self, method: str) -> dict[object, set[int]] | None:
        """Cached ``{uid: {kept local row indices}}`` for one simplify method, or None if absent."""
        path = self._dir / f"simplify_{method}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return {uid: set(group["row_index"].tolist()) for uid, group in df.groupby("uid", sort=False)}
