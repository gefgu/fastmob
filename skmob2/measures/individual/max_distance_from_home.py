from __future__ import annotations
import narwhals as nw

from typing import Any

from skmob2._core import (
    home_location_arrow,
    home_location_indexed_arrow,
    home_location_indexed_numpy,
    home_location_numpy,
    max_distance_from_point_arrow,
    max_distance_from_point_indexed_arrow,
    max_distance_from_point_indexed_numpy,
    max_distance_from_point_numpy,
)

from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _extract_hours,
    _detect_trajectory_columns,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "home": home_location_arrow,
        "home_indexed": home_location_indexed_arrow,
        "max_dist": max_distance_from_point_arrow,
        "max_dist_indexed": max_distance_from_point_indexed_arrow,
        "format_values": _arrow_result_values,
    },
    numpy_ops={
        "home": home_location_numpy,
        "home_indexed": home_location_indexed_numpy,
        "max_dist": max_distance_from_point_numpy,
        "max_dist_indexed": max_distance_from_point_indexed_numpy,
        "format_values": lambda values: values,
    },
)


def max_distance_from_home(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Return the maximum Haversine distance (km) from each user's home location.

    The maximum distance from home :math:`dh_{max}(u)` of an individual
    :math:`u` is defined as [CM2015]_:

    .. math::

        dh_{max}(u) = \\max_{1 \\leq i \\leq n_u} dist(r_i, h(u))

    where :math:`n_u` is the number of recorded points for :math:`u`,
    :math:`r_i` is a location as a :math:`(lat, lng)` pair, :math:`h(u)` is
    the home location of :math:`u` (see :func:`home_location`), and
    :math:`dist` is the Haversine distance.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    start_night : int, optional
        Hour (0–23) at which the nighttime window begins.  Forwarded to
        :func:`home_location`.  Default: 22.
    end_night : int, optional
        Hour (0–23) at which the nighttime window ends (exclusive).  Forwarded
        to :func:`home_location`.  Default: 7.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.
    presorted : bool, optional
        When True, trust that rows are already grouped by user and use the
        contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
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

    See Also
    --------
    home_location : Inferred home location from nighttime visits.
    maximum_distance : Largest single jump length per user.
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    df, hours = _extract_hours(df, datetime_col)
    lats = df.get_column(lat_col)
    lngs = df.get_column(lng_col)
    ops = _DISPATCHER.get_ops(df)
    lats_data = ops["extract_data"](lats)
    lngs_data = ops["extract_data"](lngs)
    hours_data = ops["extract_data"](hours)
    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        home_lats, home_lngs = ops["home"](lats_data, lngs_data, hours_data, ends, float(start_night), float(end_night))
        max_distances = ops["format_values"](ops["max_dist"](home_lats, home_lngs, lats_data, lngs_data, ends))
        if uid_col is None:
            return _to_native({"max_distance_from_home": max_distances}, df)
        return _to_native({uid_col: uid_values, "max_distance_from_home": max_distances}, df)

    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col)

    home_lats, home_lngs = ops["home_indexed"](lats_data, lngs_data, hours_data, indices, ends, float(start_night), float(end_night))
    max_distances = ops["format_values"](ops["max_dist_indexed"](home_lats, home_lngs, lats_data, lngs_data, indices, ends))

    if uid_col is None:
        return _to_native({"max_distance_from_home": max_distances}, df)
    return _to_native({uid_col: uid_values, "max_distance_from_home": max_distances}, df)
