from __future__ import annotations
import narwhals as nw

import math
from typing import Any

import numpy as np
from skmob2._core import (
    maximum_distance_arrow,
    maximum_distance_indexed_arrow,
    maximum_distance_indexed_numpy,
    maximum_distance_numpy,
)

from .._common import (
    _build_presorted_user_ranges,
    _build_time_ordered_user_ranges,
    _dispatch_kernel,
    _extract_timestamps_ms,
    _is_polars_backed,
    _detect_trajectory_columns,
    _prepare_trajectory,
    _to_native,
)


def maximum_distance(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
) -> Any:
    """Return the maximum distance (km) covered in a single movement for each user.

    The maximum distance is the largest Haversine distance between any two
    consecutive GPS fixes in a user's sorted trajectory.

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
    sorted:
        When True, trust that rows are already grouped by user and ordered by
        datetime within each user, then use the contiguous fast path.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "maximum_distance"]``.
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
    >>> from skmob2 import maximum_distance
    >>> result = maximum_distance(df)
    >>> print(result.round({"maximum_distance": 3}).head().to_string(index=False))
     uid  maximum_distance
       0         11294.452
       1         12804.913
       2         11286.761

    References
    ----------
    - [WTDED2015] Williams, N. E., Thomas, T. A., Dunbar, M., Eagle, N. & Dobra, A. (2015) Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data. PLOS ONE 10(7): e0133630. https://doi.org/10.1371/journal.pone.0133630
    - [LBH2012] Lu, X., Bengtsson, L. & Holme, P. (2012) Predictability of population displacement after the 2010 haiti earthquake. Proceedings of the National Academy of Sciences 109 (29) 11576-11581; https://doi.org/10.1073/pnas.1203882109
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    use_arrow = _is_polars_backed(df)
    if sorted:
        uid_values, ranges = _build_presorted_user_ranges(df, uid_col)
        max_distances = _dispatch_kernel(
            maximum_distance_numpy,
            maximum_distance_arrow,
            [df.get_column(lat_col), df.get_column(lng_col)],
            ranges,
            use_arrow=use_arrow,
        )
        if uid_col is None:
            if len(df) < 2:
                return _to_native({"maximum_distance": [math.nan]}, df)
            return _to_native({"maximum_distance": max_distances}, df)

        max_distances = np.asarray(max_distances, dtype=float)
        short_mask = np.fromiter((end - start < 2 for start, end in ranges), dtype=bool, count=len(ranges))
        max_distances[short_mask] = math.nan
        return _to_native({uid_col: uid_values, "maximum_distance": max_distances}, df)

    timestamps = _extract_timestamps_ms(df, datetime_col)
    uid_values, indices, ends = _build_time_ordered_user_ranges(
        df, uid_col, datetime_col, timestamps, use_arrow=use_arrow
    )
    max_distances = _dispatch_kernel(
        maximum_distance_indexed_numpy,
        maximum_distance_indexed_arrow,
        [df.get_column(lat_col), df.get_column(lng_col)],
        indices,
        ends,
        use_arrow=use_arrow,
    )

    if uid_col is None:
        if len(df) < 2:
            return _to_native({"maximum_distance": [math.nan]}, df)
        return _to_native({"maximum_distance": max_distances}, df)

    max_distances = np.asarray(max_distances, dtype=float)
    ends_array = np.asarray(ends, dtype=np.uintp)
    starts_array = np.empty_like(ends_array)
    if len(ends_array):
        starts_array[0] = 0
        starts_array[1:] = ends_array[:-1]
    short_mask = ends_array - starts_array < 2
    max_distances[short_mask] = math.nan
    return _to_native({uid_col: uid_values, "maximum_distance": max_distances}, df)
