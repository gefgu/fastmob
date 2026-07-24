"""Staypoints — validated, chronologically-bounded stop-location dataframe.

The second level of the Positionfixes -> Staypoints -> Triplegs -> Trips ->
Tours hierarchy (see `fastmob/core/positionfixes_dataframe.py`). Built via
`Positionfixes.generate_staypoints()`, which wraps the existing
`fastmob.preprocessing.stay_locations` Rust-backed detector -- no new Rust
kernel is needed for this level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw

from ..measures._common import LAT_CANDIDATES, LNG_CANDIDATES, _pick_existing_column
from ._hierarchy_common import _detect_interval_columns
from .base import BaseDataFrame

if TYPE_CHECKING:
    from .locations_dataframe import Locations


class Staypoints(BaseDataFrame):
    """One row per detected stop: a place a user stayed for a while.

    Parameters
    ----------
    df : DataFrame-like
        Source data; any Narwhals-compatible eager backend.
    uid_col, lat_col, lng_col, started_at_col, finished_at_col : str, optional
        Explicit column name overrides; auto-detected when None.
    validate : bool, optional
        When True (default), check that required columns are present and
        that ``finished_at >= started_at`` for every row.
    """

    def __init__(
        self,
        df: Any,
        uid_col: str | None = None,
        lat_col: str | None = None,
        lng_col: str | None = None,
        started_at_col: str | None = None,
        finished_at_col: str | None = None,
        parameters: dict | None = None,
        validate: bool = True,
    ):
        super().__init__(df)
        nw_df = nw.from_native(df, eager_only=True)
        columns = nw_df.columns

        if lat_col is None:
            lat_col = _pick_existing_column(columns, LAT_CANDIDATES)
        if lng_col is None:
            lng_col = _pick_existing_column(columns, LNG_CANDIDATES)
        uid_col, started_at_col, finished_at_col = _detect_interval_columns(
            columns, uid_col, started_at_col, finished_at_col
        )

        self.uid_col = uid_col
        self.lat_col = lat_col
        self.lng_col = lng_col
        self.started_at_col = started_at_col
        self.finished_at_col = finished_at_col
        self.parameters = {} if parameters is None else dict(parameters)

        if validate:
            self.validate(nw_df, started_at_col, finished_at_col)

    @staticmethod
    def validate(df: Any, started_at_col: str, finished_at_col: str) -> None:
        """Check required columns are present and ``finished_at >= started_at``."""
        nw_df = nw.from_native(df, eager_only=True) if not isinstance(df, nw.DataFrame) else df
        for col in (started_at_col, finished_at_col):
            if col not in nw_df.columns:
                raise ValueError(f"Staypoints requires a '{col}' column")
        if len(nw_df) == 0:
            return
        bad_rows = nw_df.filter(nw.col(finished_at_col) < nw.col(started_at_col))
        if len(bad_rows) > 0:
            raise ValueError(
                f"Staypoints requires finished_at >= started_at for every row ({len(bad_rows)} violating row(s))"
            )

    def generate_locations(
        self,
        epsilon_km: float = 0.1,
        min_samples: int = 1,
        agg_level: str = "user",
    ) -> tuple[Locations, Staypoints]:
        """Cluster staypoints spatially into recurring `Locations`.

        Parameters
        ----------
        epsilon_km : float, optional
            DBSCAN neighborhood radius, in km. Default ``0.1``.
        min_samples : int, optional
            DBSCAN minimum samples per cluster. Default ``1``.
        agg_level : str, optional
            Only ``"user"`` (cluster each user's own staypoints
            independently) is implemented; ``"dataset"``-level (shared
            cross-user locations) is a planned follow-up.

        Returns
        -------
        tuple[Locations, Staypoints]
            The detected locations, and a copy of ``self`` with a new
            ``location_id`` column (null where a staypoint didn't join any
            recurring location).
        """
        if agg_level != "user":
            raise NotImplementedError(
                f"generate_locations(agg_level={agg_level!r}) is not implemented yet; only 'user' is supported"
            )

        from ..preprocessing import cluster
        from .locations_dataframe import Locations

        clustered = cluster(
            self.df,
            cluster_radius_km=epsilon_km,
            min_samples=min_samples,
            datetime_col=self.started_at_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )
        nwc = nw.from_native(clustered, eager_only=True)
        nwc = nwc.with_columns(
            nw.when(nw.col("cluster") >= 0).then(nw.col("cluster")).otherwise(None).alias("location_id")
        ).drop("cluster")

        group_cols = [self.uid_col, "location_id"] if self.uid_col else ["location_id"]
        located = nwc.filter(~nw.col("location_id").is_null())
        locations_df = located.group_by(group_cols).agg(
            nw.col(self.lat_col).mean().alias("center_lat"),
            nw.col(self.lng_col).mean().alias("center_lng"),
            nw.len().alias("n_staypoints"),
        )

        staypoints_with_location = Staypoints(
            nwc.to_native(),
            uid_col=self.uid_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            started_at_col=self.started_at_col,
            finished_at_col=self.finished_at_col,
            parameters=self.parameters,
        )
        locations = Locations(locations_df.to_native(), uid_col=self.uid_col)
        return locations, staypoints_with_location
