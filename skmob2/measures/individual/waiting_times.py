from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    waiting_times_arrow,
    waiting_times_flat_arrow,
    waiting_times_flat_numpy,
    waiting_times_indexed_arrow,
    waiting_times_indexed_flat_arrow,
    waiting_times_indexed_flat_numpy,
    waiting_times_indexed_numpy,
    waiting_times_numpy,
)

from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _build_presorted_user_ranges,
    _build_time_ordered_user_ranges,
    _extract_timestamps_s,
    _detect_trajectory_columns,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "indexed": waiting_times_indexed_arrow,
        "indexed_flat": waiting_times_indexed_flat_arrow,
        "presorted": waiting_times_arrow,
        "presorted_flat": waiting_times_flat_arrow,
    },
    numpy_ops={
        "indexed": waiting_times_indexed_numpy,
        "indexed_flat": waiting_times_indexed_flat_numpy,
        "presorted": waiting_times_numpy,
        "presorted_flat": waiting_times_flat_numpy,
    },
)


def _route_indexed_waiting_times(
    ops: dict,
    timestamps_data: Any,
    indices: np.ndarray,
    ends: np.ndarray,
    *,
    merge: bool,
) -> Any:
    if merge:
        return ops["indexed_flat"](timestamps_data, indices, ends)
    return ops["indexed"](timestamps_data, indices, ends)


def _route_presorted_waiting_times(
    ops: dict,
    timestamps_data: Any,
    ranges: list[tuple[int, int]],
    *,
    merge: bool,
) -> Any:
    if merge:
        return ops["presorted_flat"](timestamps_data, ranges)
    return ops["presorted"](timestamps_data, ranges)


def waiting_times(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
) -> Any:
    """Return the waiting times (seconds) between consecutive GPS fixes for each user.

    The waiting time at position i is the number of seconds elapsed between
    trajectory point i and point i+1, within the same user's sorted trajectory.
    Users with fewer than 2 points receive an empty list.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge:
        When ``True``, return a single flat Python list of all waiting times
        across all users concatenated together.  When ``False`` (default),
        return a DataFrame with one row per user.
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
    DataFrame or list
        When ``merge=False``: one row per user with columns
        ``[uid_col, "waiting_times"]``; each cell is a list of floats (seconds).
        When ``merge=True``: a flat Python ``list[float]`` of all waiting times.
        The returned DataFrame backend matches the input backend.



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
    >>> from skmob2 import waiting_times
    >>> result = waiting_times(df)
    >>> preview = result.assign(n_waiting_times=result["waiting_times"].str.len())
    >>> print(preview[["uid", "n_waiting_times"]].head().to_string(index=False))
     uid  n_waiting_times
       0             2098
       1             1209
       2             1690

    References
    ----------
    - [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4
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

    ops = _DISPATCHER.get_ops(df)
    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"
    timestamps_s = _extract_timestamps_s(df, datetime_col)
    timestamps_data = ops["extract_data"](timestamps_s)
    if sorted:
        uid_values, ranges = _build_presorted_user_ranges(df, uid_col)
        wt_values = _route_presorted_waiting_times(ops, timestamps_data, ranges, merge=merge)
        if merge:
            return wt_values
        if uid_col is None:
            return _to_native({"waiting_times": wt_values}, df)
        return _to_native({uid_col: uid_values, "waiting_times": wt_values}, df)

    uid_values, indices, ends = _build_time_ordered_user_ranges(
        df, uid_col, datetime_col, timestamps_s, use_arrow=use_arrow
    )
    wt_values = _route_indexed_waiting_times(ops, timestamps_data, indices, ends, merge=merge)

    if merge:
        return wt_values
    if uid_col is None:
        return _to_native({"waiting_times": wt_values}, df)
    return _to_native({uid_col: uid_values, "waiting_times": wt_values}, df)
