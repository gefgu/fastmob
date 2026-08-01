from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import total_distance_indexed, total_distance_presorted
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _arrow_result_values,
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _extract_timestamps_ms,
    _to_native,
)

_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def distance_straight_line(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Return the total trajectory length (km) for each user.

    The distance straight line :math:`d_{SL}` travelled by an individual
    :math:`u` is the sum of Haversine distances between consecutive GPS fixes
    in the time-ordered trajectory [WTDED2015]_:

    .. math::

        d_{SL}(u) = \\sum_{j=2}^{n_u} dist(r_{j-1}, r_j)

    where :math:`n_u` is the number of recorded points for :math:`u`,
    :math:`r_j` is the :math:`j`-th point as a :math:`(lat, lng)` pair, and
    :math:`dist` is the Haversine distance between two points.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.
    presorted : bool, optional
        When True, trust that rows are already grouped by user and ordered by
        datetime within each user, then use the contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[uid_col, "distance_straight_line"]``.
        Distance values are in kilometres.
        The returned backend matches the input backend.


    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from fastmob import distance_straight_line
    >>> result = distance_straight_line(df)
    >>> print(result.round({"distance_straight_line": 3}).head().to_string(index=False))
     uid  distance_straight_line
       0              374531.472
       1              774347.886
       2               86036.423

    References
    ----------
    - [WTDED2015] Williams, N. E., Thomas, T. A., Dunbar, M., Eagle, N. & Dobra, A. (2015) Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data. PLOS ONE 10(7): e0133630. https://doi.org/10.1371/journal.pone.0133630

    See Also
    --------
    jump_lengths : Individual jump distances between consecutive points.
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

    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        distances = _arrow_result_values(total_distance_presorted(lats_data, lngs_data, ends))
        if uid_col is None:
            return _to_native({"distance_straight_line": distances}, df)
        return _to_native({uid_col: uid_values, "distance_straight_line": distances}, df)

    timestamps = _extract_timestamps_ms(df, datetime_col)
    uid_values, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps)
    distances = _arrow_result_values(total_distance_indexed(lats_data, lngs_data, indices, ends))

    if uid_col is None:
        return _to_native({"distance_straight_line": distances}, df)
    return _to_native({uid_col: uid_values, "distance_straight_line": distances}, df)
