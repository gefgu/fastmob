"""Positionfixes — the raw-GPS-fix level of the trajectory hierarchy.

`TrajDataFrame` already *is* this level (one row per raw GPS fix, a single
`datetime`, `lat`/`lng`, optional `uid`). `Positionfixes` is a stable,
semantically-named alias with room to grow hierarchy-specific convenience
methods (`generate_staypoints`, `generate_triplegs`) without changing
`TrajDataFrame` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw

from .trajectory_dataframe import TrajDataFrame

if TYPE_CHECKING:
    from .staypoints_dataframe import Staypoints
    from .triplegs_dataframe import Triplegs


class Positionfixes(TrajDataFrame):
    """Semantic alias for `TrajDataFrame` at the base of the
    Positionfixes -> Staypoints -> Triplegs -> Trips -> Tours hierarchy.
    """

    def generate_staypoints(self, **kwargs: Any) -> Staypoints:
        """Detect stop locations, returning them as a typed `Staypoints` level.

        Thin wrapper around `fastmob.preprocessing.stay_locations`: renames
        its ``datetime``/``leaving_datetime`` output columns to
        ``started_at``/``finished_at`` and records the stop-detection
        parameters used (so `generate_triplegs` can reuse them by default).

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`fastmob.preprocessing.stay_locations`
            (``minutes_for_a_stop``, ``spatial_radius_km``, etc.).

        Returns
        -------
        Staypoints

        Examples
        --------
        >>> import pandas as pd
        >>> from fastmob import Positionfixes
        >>> base = pd.Timestamp("2024-01-01")
        >>> rows = [
        ...     {"uid": "u1", "datetime": base + pd.Timedelta(minutes=m), "lat": 0.0, "lng": 0.0}
        ...     for m in range(0, 31, 5)
        ... ] + [
        ...     {"uid": "u1", "datetime": base + pd.Timedelta(minutes=30 + i), "lat": 0.0, "lng": 0.01 * i}
        ...     for i in range(1, 11)
        ... ] + [
        ...     {"uid": "u1", "datetime": base + pd.Timedelta(minutes=41 + m), "lat": 0.0, "lng": 0.1}
        ...     for m in range(0, 31, 5)
        ... ]
        >>> fixes = Positionfixes(pd.DataFrame(rows))
        >>> stays = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
        >>> stays.df[["staypoint_id", "lng"]].to_dict("records")
        [{'staypoint_id': 0, 'lng': 0.0}, {'staypoint_id': 1, 'lng': 0.1}]
        """
        from ..preprocessing import stay_locations
        from .staypoints_dataframe import Staypoints

        stops = stay_locations(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            leaving_time=True,
            **kwargs,
        )
        stops_nw = nw.from_native(stops, eager_only=True).rename(
            {self.datetime_col: "started_at", "leaving_datetime": "finished_at"}
        )
        # A stable identity for each staypoint, assigned once here and
        # carried through as an ordinary data column by every later
        # operation (generate_user_locations' cluster() resort, create_activity_flag,
        # ...) -- Trips.generate_trips needs it to record which staypoint
        # brackets each trip.
        stops_nw = stops_nw.with_row_index("staypoint_id")
        return Staypoints(
            stops_nw.to_native(),
            uid_col=self.uid_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            started_at_col="started_at",
            finished_at_col="finished_at",
            parameters=dict(kwargs),
        )

    def generate_triplegs(
        self,
        staypoints: Staypoints,
        gap_threshold_min: float = 15.0,
        **stop_kwargs: Any,
    ) -> Triplegs:
        """Derive movement segments (triplegs) between staypoints.

        Parameters
        ----------
        staypoints : Staypoints
            Staypoints previously generated from this same trajectory (ideally
            via :meth:`generate_staypoints`, so stop-detection parameters
            match).
        gap_threshold_min : float, optional
            Reserved for future gap-based tripleg splitting; unused by the
            current ``"between_staypoints"`` method. Default ``15.0``.
        **stop_kwargs
            Stop-detection parameter overrides forwarded to
            :func:`fastmob.preprocessing.segment` (``stop_radius_km``,
            ``minutes_for_a_stop``, ...). Defaults to the parameters recorded
            on ``staypoints`` (from :meth:`generate_staypoints`) when not
            given.

        Returns
        -------
        Triplegs

        Examples
        --------
        >>> staypoints = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
        >>> triplegs = fixes.generate_triplegs(staypoints)
        >>> triplegs.df[["tripleg_id", "duration_s"]].to_dict("records")
        [{'tripleg_id': 1, 'duration_s': 540.0}]
        """
        from .triplegs_dataframe import Triplegs

        return Triplegs.from_positionfixes(self, staypoints, gap_threshold_min=gap_threshold_min, **stop_kwargs)
