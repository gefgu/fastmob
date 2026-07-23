"""Utilities for loading cached PTRAIL reference results.

The cache is populated once by running:
    bash scripts/populate_ptrail_cache.sh

in the .venv-ptrail environment. The resulting files in
tests/shared/ptrail_reference/ are committed to git so the normal .venv can
run comparison tests without ptrail installed.

Cache schema (see the project plan's "Cache schema per capability area"):
    input.parquet
        Standardised pandas DataFrame fed to PTRAIL, with an extra
        ``row_index`` column giving each row's 0-based position within its
        user's chronologically-sorted slice (the same local numbering
        fastmob's ``filter()`` mask-as-indices uses for outlier methods).
    outlier_hampel.parquet
        Columns ``uid`` and ``row_index``: one row per (user, kept local
        row index) pair that PTRAIL's ``Filters.hampel_outlier_detection``
        kept — i.e. did *not* flag as an outlier on the ``Speed`` kinematic
        feature column (see ``tests/populate_ptrail_cache.py``'s module
        docstring for the version-compatibility note this depends on).
    interpolate_<method>.parquet
        Columns ``uid``, ``datetime``, ``lat``, ``lon``: every row PTRAIL's
        real per-user interpolation helper (``Helpers.linear_help`` /
        ``cubic_help`` / ``kinematic_help``, called directly — see
        ``tests/populate_ptrail_cache.py``) produced for that user, sorted
        chronologically (original points plus any inserted point). One file
        per entry in ``INTERPOLATE_METHODS`` (``"linear"``, ``"cubic"``,
        ``"kinematic"``).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

_REFERENCE_DIR = Path(__file__).parent / "ptrail_reference"


class PtrailReferenceDataset:
    """Cached PTRAIL outputs for one dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    @functools.cached_property
    def input_df(self) -> pd.DataFrame:
        """Standardised pandas DataFrame fed to PTRAIL (uid, datetime, lat, lng, row_index)."""
        return pd.read_parquet(self._dir / "input.parquet")

    def keep_mask(self, method: str) -> dict[object, set[int]] | None:
        """Cached ``{uid: {kept local row indices}}`` for one outlier method, or None if absent."""
        path = self._dir / f"outlier_{method}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return {uid: set(group["row_index"].tolist()) for uid, group in df.groupby("uid", sort=False)}

    def interpolated_positions(self, method: str) -> dict[object, pd.DataFrame] | None:
        """Cached ``{uid: DataFrame(datetime, lat, lon)}`` for one interpolation
        method (``"linear"``, ``"cubic"``, or ``"kinematic"``), sorted
        chronologically per user (original points plus any point PTRAIL
        inserted), or None if the cache file is absent.
        """
        path = self._dir / f"interpolate_{method}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return {
            uid: group[["datetime", "lat", "lon"]].reset_index(drop=True)
            for uid, group in df.groupby("uid", sort=False)
        }
