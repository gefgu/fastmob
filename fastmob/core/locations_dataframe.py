"""Locations — spatially-aggregated recurring stop locations.

Built via `Staypoints.generate_locations()`, which wraps the existing
`fastmob.preprocessing.cluster` DBSCAN stop-clustering -- no new Rust kernel
is needed for this level.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .base import BaseDataFrame


class Locations(BaseDataFrame):
    """One row per detected recurring location (a spatial cluster of staypoints).

    Parameters
    ----------
    df : DataFrame-like
        Source data; any Narwhals-compatible eager backend.
    uid_col : str, optional
        User-ID column name. ``None`` when locations were clustered across
        all users at once (not yet supported by `generate_locations`).
    location_id_col, center_lat_col, center_lng_col : str, optional
        Column name overrides.
    validate : bool, optional
        When True (default), check that required columns are present.
    """

    def __init__(
        self,
        df: Any,
        uid_col: str | None = None,
        location_id_col: str = "location_id",
        center_lat_col: str = "center_lat",
        center_lng_col: str = "center_lng",
        validate: bool = True,
    ):
        super().__init__(df)
        self.uid_col = uid_col
        self.location_id_col = location_id_col
        self.center_lat_col = center_lat_col
        self.center_lng_col = center_lng_col

        if validate:
            nw_df = nw.from_native(df, eager_only=True)
            required = [location_id_col, center_lat_col, center_lng_col]
            missing = [col for col in required if col not in nw_df.columns]
            if missing:
                raise ValueError(f"Locations is missing required columns: {missing}")
