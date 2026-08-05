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

from fastmob.utils._common import (
    LAT_CANDIDATES,
    LNG_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _detect_required_column,
    _pick_existing_column,
)

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
    def validate(df: Any, started_at_col: str, finished_at_col: str | None = None) -> None:
        """Check the start column and, when present, interval ordering."""
        nw_df = nw.from_native(df, eager_only=True) if not isinstance(df, nw.DataFrame) else df
        columns = [started_at_col] if finished_at_col is None else [started_at_col, finished_at_col]
        for col in columns:
            if col not in nw_df.columns:
                raise ValueError(f"Staypoints requires a '{col}' column")
        if finished_at_col is None or len(nw_df) == 0:
            return
        bad_rows = nw_df.filter(nw.col(finished_at_col) < nw.col(started_at_col))
        if len(bad_rows) > 0:
            raise ValueError(
                f"Staypoints requires finished_at >= started_at for every row ({len(bad_rows)} violating row(s))"
            )

    @staticmethod
    def resolve_dataframe(
        staypoints: Any | Staypoints,
        *,
        user_id_col: str | None = None,
        timestamp_col: str | None = None,
        lat_col: str | None = None,
        lng_col: str | None = None,
        require_coordinates: bool = False,
    ) -> tuple[nw.DataFrame, str, str, str | None, str | None]:
        """Normalize staypoint input, resolve measure columns, and validate starts.

        ``Staypoints`` metadata takes precedence over candidate detection. Raw
        dataframe inputs stay supported for measures whose historical API did
        not require an interval end column.
        """
        if isinstance(staypoints, Staypoints):
            user_id_col = user_id_col or staypoints.uid_col
            timestamp_col = timestamp_col or staypoints.started_at_col
            lat_col = lat_col or staypoints.lat_col
            lng_col = lng_col or staypoints.lng_col
            staypoints = staypoints.df

        df = nw.from_native(staypoints, eager_only=True)
        user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES)
        timestamp_col = _detect_required_column(df, timestamp_col, TIMESTAMP_CANDIDATES)
        if require_coordinates:
            lat_col = _detect_required_column(df, lat_col, LAT_CANDIDATES)
            lng_col = _detect_required_column(df, lng_col, LNG_CANDIDATES)
        Staypoints.validate(df, timestamp_col)
        return df, user_id_col, timestamp_col, lat_col, lng_col

    def create_activity_flag(self, method: str = "time_threshold", time_threshold_min: float = 15.0) -> Staypoints:
        """Flag each staypoint as a genuine "activity" by dwell time.

        See :func:`fastmob.preprocessing.create_activity_flag`.
        """
        from ..preprocessing import create_activity_flag

        return create_activity_flag(self, method=method, time_threshold_min=time_threshold_min)

    def generate_user_locations(
        self,
        epsilon_km: float = 0.1,
        min_samples: int = 1,
    ) -> tuple[Locations, Staypoints]:
        """Cluster each user's staypoints into recurring locations.

        Parameters
        ----------
        epsilon_km : float, optional
            Approximate spatial scale, in km, used to select an H3 grid
            resolution. Default ``0.1``.
        min_samples : int, optional
            Minimum staypoints in an H3 cell for it to be active. Default
            ``1``.
        Returns
        -------
        tuple[Locations, Staypoints]
            The detected locations, and a copy of ``self`` with a new
            ``location_id`` column (null where a staypoint didn't join any
            recurring location).
        """
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
        locations = Locations(locations_df.to_native(), uid_col=self.uid_col, scope="user", scheme="cluster")
        return locations, staypoints_with_location

    def generate_global_locations(self, h3_resolution: int = 9) -> tuple[Locations, Staypoints]:
        """Assign staypoints to shared H3 locations and return their catalogue.

        The returned ``Locations`` has global scope and one row per occupied
        H3 cell; the returned staypoints carry that cell as ``location_id``.
        """
        if isinstance(h3_resolution, bool) or not isinstance(h3_resolution, int) or not 0 <= h3_resolution <= 15:
            raise ValueError("h3_resolution must be an integer between 0 and 15")
        import pyarrow as pa

        from fastmob._core import h3_to_latlng_arrow, latlng_to_h3_arrow

        from .locations_dataframe import Locations

        nwc = nw.from_native(self.df, eager_only=True)
        cells = latlng_to_h3_arrow(
            nwc.get_column(self.lat_col).to_arrow(), nwc.get_column(self.lng_col).to_arrow(), h3_resolution
        )
        center_lats, center_lngs = h3_to_latlng_arrow(cells)
        arrow = nw.from_arrow(
            pa.table({"location_id": cells, "__center_lat__": center_lats, "__center_lng__": center_lngs}),
            backend=nwc.implementation,
        )
        assigned = nwc.with_columns(
            arrow.get_column("location_id").alias("location_id"),
            arrow.get_column("__center_lat__").alias("__center_lat__"),
            arrow.get_column("__center_lng__").alias("__center_lng__"),
        )
        locations_df = (
            assigned.drop_nulls(subset=["location_id"])
            .group_by("location_id")
            .agg(
                nw.col("__center_lat__").first().alias("center_lat"),
                nw.col("__center_lng__").first().alias("center_lng"),
                nw.len().alias("n_staypoints"),
            )
        )
        assigned = assigned.drop("__center_lat__", "__center_lng__")
        locations = Locations(locations_df.to_native(), scope="global", scheme="h3")
        return locations, Staypoints(
            assigned.to_native(),
            uid_col=self.uid_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            started_at_col=self.started_at_col,
            finished_at_col=self.finished_at_col,
            parameters=self.parameters,
        )

    def associate_global_locations(self, locations: Locations, location_id_col: str = "location_id") -> Staypoints:
        """Validate preassigned exact global IDs against a location catalogue."""
        if locations.scope != "global":
            raise ValueError("associate_global_locations requires global Locations")
        df = nw.from_native(self.df, eager_only=True)
        if location_id_col not in df.columns:
            raise ValueError(f"Staypoints is missing global location-ID column {location_id_col!r}")
        known = set(nw.from_native(locations.df, eager_only=True).get_column(locations.location_id_col).to_list())
        assigned = set(df.get_column(location_id_col).drop_nulls().to_list())
        unknown = assigned - known
        if unknown:
            raise ValueError("Staypoints contains location IDs absent from the global Locations catalogue")
        if location_id_col != "location_id":
            df = df.rename({location_id_col: "location_id"})
        return Staypoints(
            df.to_native(),
            uid_col=self.uid_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            started_at_col=self.started_at_col,
            finished_at_col=self.finished_at_col,
            parameters=self.parameters,
        )

    def build_stvd(self, locations: Locations, **kwargs: Any) -> Any:
        """Aggregate these staypoints against a global Locations catalogue into an STVD frame.

        See :func:`fastmob.measures.collective.build_stvd`.
        """
        from ..measures.collective.stvd import build_stvd as _build_stvd

        return _build_stvd(self, locations, **kwargs)

    def generate_daily_motifs(
        self,
        locations: Locations,
        *,
        presorted: bool = False,
    ) -> Any:
        """Compute one home-anchored mobility motif per user and day.

        See :func:`fastmob.measures.individual.motifs.daily_motifs_from_staypoints`.
        """
        from ..measures.individual.motifs import daily_motifs_from_staypoints

        return daily_motifs_from_staypoints(self, locations, presorted=presorted)
