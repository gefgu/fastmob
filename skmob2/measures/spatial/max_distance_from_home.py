from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    home_location_indexed_arrow,
    home_location_indexed_numpy,
    max_distance_from_point_indexed_arrow,
    max_distance_from_point_indexed_numpy,
)

from .._common import _arrow_result_values, _build_indexed_user_ranges_fast, _is_polars_backed, _prepare_trajectory


def max_distance_from_home(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the maximum Haversine distance (km) from each user's home location.

    The home location is computed via :func:`home_location` using the
    nighttime-window heuristic.  The distance is the maximum Haversine
    distance from that home point to any trajectory point for the user.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    start_night:
        Hour (0–23) at which the nighttime window begins.  Forwarded to
        :func:`home_location`.  Default: 22.
    end_night:
        Hour (0–23) at which the nighttime window ends (exclusive).  Forwarded
        to :func:`home_location`.  Default: 7.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "max_distance_from_home"]``.
        Distance values are in kilometres.  The returned backend matches the
        input backend.



    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from skmob2 import max_distance_from_home
    >>> result = max_distance_from_home(df)
    >>> print(result.round({"max_distance_from_home": 3}).head().to_string(index=False))
     uid  max_distance_from_home
       0               11286.959
       1               12800.565
       2               11282.764

    References
    ----------
    - [CM2015] Canzian, L. & Musolesi, M. (2015) Trajectories of depression: unobtrusive monitoring of depressive states by means of smartphone mobility traces analysis. Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing, 1293-1304, https://dl.acm.org/citation.cfm?id=2805845

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    df = df.with_columns(nw.col(datetime_col).dt.hour().cast(nw.Float64).alias("__hour__"))
    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    hours = df.get_column("__hour__")
    use_arrow = _is_polars_backed(df)
    uid_values, indices, starts, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)

    if use_arrow:
        home_lats, home_lngs = home_location_indexed_arrow(
            lats_full.to_arrow(),
            lngs_full.to_arrow(),
            hours.to_arrow(),
            indices,
            starts,
            ends,
            float(start_night),
            float(end_night),
        )
        max_distances = _arrow_result_values(
            max_distance_from_point_indexed_arrow(
                home_lats,
                home_lngs,
                lats_full.to_arrow(),
                lngs_full.to_arrow(),
                indices,
                starts,
                ends,
            )
        )
    else:
        home_lats, home_lngs = home_location_indexed_numpy(
            lats_full.to_numpy(),
            lngs_full.to_numpy(),
            hours.to_numpy(),
            indices,
            starts,
            ends,
            float(start_night),
            float(end_night),
        )
        max_distances = max_distance_from_point_indexed_numpy(
            home_lats,
            home_lngs,
            lats_full.to_numpy(),
            lngs_full.to_numpy(),
            indices,
            starts,
            ends,
        )

    if uid_col is None:
        return nw.from_dict({"max_distance_from_home": max_distances}, backend=df.implementation).to_native()

    return nw.from_dict(
        {uid_col: uid_values, "max_distance_from_home": max_distances},
        backend=df.implementation,
    ).to_native()
