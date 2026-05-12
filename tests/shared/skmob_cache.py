"""Utilities for loading cached skmob reference results.

The cache is populated once by running:
    bash tests/populate_skmob_cache.sh

in the .venv-skmob environment (Python 3.10 + scikit-mobility 1.3.1).
The resulting files in tests/shared/skmob_reference/ are committed to git
so the normal .venv can run comparison tests without skmob installed.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pandas as pd

_REFERENCE_DIR = Path(__file__).parent / "skmob_reference"


class SkmobReferenceDataset:
    """Cached skmob outputs for one dataset."""

    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        self._dir = _REFERENCE_DIR / dataset_name

    @functools.cached_property
    def input_df(self) -> pd.DataFrame:
        """Standardised pandas DataFrame that was fed to skmob (uid, lat, lng, datetime)."""
        return pd.read_parquet(self._dir / "input.parquet")

    def result(self, measure: str) -> pd.DataFrame | None:
        """Cached skmob result DataFrame, or None if not present."""
        path = self._dir / f"{measure}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def row_count(self, measure: str) -> int | None:
        """Cached row count saved for preprocessing-only comparisons."""
        path = self._dir / f"{measure}_count.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())["count"]
