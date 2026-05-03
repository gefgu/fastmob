from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    total_distance_indexed_arrow,
    total_distance_indexed_numpy,
)

from .._common import (
    _arrow_result_values,
    _as_index_array,
    _build_time_ordered_user_ranges,
    _is_polars_backed,
    _prepare_trajectory,
    _ranges_to_starts_ends,
    _route_two_series_kernel,
)


def distance_straight_line(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total trajectory length (km) for each user.

    The total distance (called "distance straight line" in skmob) is the sum
    of Haversine distances between all consecutive GPS fixes in a user's sorted
    trajectory.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per user with columns ``[uid_col, "distance_straight_line"]``.
        Distance values are in kilometres.
        The returned backend matches the input backend.



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
    >>> from skmob2 import distance_straight_line
    >>> result = distance_straight_line(df)
    >>> print(result.round({"distance_straight_line": 3}).head().to_string(index=False))
     uid  distance_straight_line
       0              374531.472
       1              774347.886
       2               86036.423

    References
    ----------
    - [WTDED2015] Williams, N. E., Thomas, T. A., Dunbar, M., Eagle, N. & Dobra, A. (2015) Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data. PLOS ONE 10(7): e0133630. https://doi.org/10.1371/journal.pone.0133630

    @usedBy
        skmob2.measures.spatial.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    timestamps = df.with_columns((nw.col(datetime_col).dt.timestamp("ms").cast(nw.Float64)).alias("__ts_ms__")).get_column(
        "__ts_ms__"
    )
    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)
    uid_values, indices, ranges = _build_time_ordered_user_ranges(df, uid_col, datetime_col, timestamps, use_arrow=use_arrow)
    starts, ends = _ranges_to_starts_ends(ranges)
    index_array = _as_index_array(indices)

    if uid_col is None:
        values = _route_two_series_kernel(lats_full, lngs_full, total_distance_indexed_numpy, total_distance_indexed_arrow, index_array, starts, ends, use_arrow=use_arrow)
        if use_arrow:
            values = _arrow_result_values(values)
        return nw.from_dict(
            {"distance_straight_line": values},
            backend=df.implementation,
        ).to_native()

    total_distances = _route_two_series_kernel(lats_full, lngs_full, total_distance_indexed_numpy, total_distance_indexed_arrow, index_array, starts, ends, use_arrow=use_arrow)
    if use_arrow:
        total_distances = _arrow_result_values(total_distances)

    return nw.from_dict(
        {uid_col: uid_values, "distance_straight_line": total_distances},
        backend=df.implementation,
    ).to_native()
