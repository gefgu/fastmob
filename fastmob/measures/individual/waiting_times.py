from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob._core import (
    waiting_times_indexed,
    waiting_times_indexed_flat,
    waiting_times_presorted,
    waiting_times_presorted_flat,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _as_arrow,
    _build_user_ranges_auto,
    _detect_trajectory_columns,
    _extract_timestamp_arrow,
    _extract_timestamps,
    _grouped_arrow_values,
    _to_native,
)

# Kept for the elapsed-time kernel input below and the merge=True output
# contract shim, matching jump_lengths.py's precedent. Ordering receives the
# original Arrow timestamp array separately.
_DISPATCHER = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def waiting_times(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the waiting times (seconds) between consecutive GPS fixes for each user.

    A waiting time (or inter-time) :math:`\\Delta t` is the elapsed time
    between two consecutive trajectory points of individual :math:`u`
    [SKWB2010]_ [PF2018]_:

    .. math::

        \\Delta t = |t(r_i) - t(r_{i+1})|

    where :math:`r_i` and :math:`r_{i+1}` are two consecutive points in the
    time-ordered trajectory and :math:`t(r)` is the time when :math:`u`
    visited point :math:`r`.  Users with fewer than 2 points receive an empty
    list.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge : bool, optional
        When ``True``, return flat waiting times across all users using a
        backend-appropriate array object. When ``False`` (default), return a
        DataFrame with one row per user.
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
    pandas.DataFrame or polars.DataFrame or array-like
        When ``merge=False``: one row per user with columns
        ``[uid_col, "waiting_times"]``; each cell is an array-like sequence of
        floats (seconds).
        When ``merge=True``: flat waiting times as a NumPy array for NumPy-backed
        inputs or a PyArrow array for Arrow-backed inputs.
        The returned DataFrame backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.data.BRIGHTKITE_SAMPLE
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
    >>> from fastmob import waiting_times
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
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    datetime_col, _lat_col, _lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    ops = _DISPATCHER.get_ops(df)
    timestamps = _extract_timestamps(df, datetime_col)
    timestamp_arrow = _extract_timestamp_arrow(df, datetime_col)
    timestamps_data = ops["extract_data"](timestamps)
    is_numpy_backend = _DISPATCHER.get_backend_key(df) == "numpy"
    uid_values, indices, ends = _build_user_ranges_auto(df, uid_col, timestamp_arrow)
    if indices is None:
        if merge:
            flat = _as_arrow(waiting_times_presorted_flat(timestamps_data, ends))
            return flat.to_numpy(zero_copy_only=False) if is_numpy_backend else flat
        value_starts, value_ends, flat_values = waiting_times_presorted(timestamps_data, ends)
        flat_values = _as_arrow(flat_values)
        wt_values = _grouped_arrow_values(value_starts, value_ends, flat_values, value_offsets=True)
        if uid_col is None:
            return _to_native({"waiting_times": wt_values}, df)
        return _to_native({uid_col: uid_values, "waiting_times": wt_values}, df)

    if merge:
        flat = _as_arrow(waiting_times_indexed_flat(timestamps_data, indices, ends))
        return flat.to_numpy(zero_copy_only=False) if is_numpy_backend else flat

    value_starts, value_ends, flat_values = waiting_times_indexed(timestamps_data, indices, ends)
    flat_values = _as_arrow(flat_values)
    wt_values = _grouped_arrow_values(value_starts, value_ends, flat_values, value_offsets=True)
    if uid_col is None:
        return _to_native({"waiting_times": wt_values}, df)
    return _to_native({uid_col: uid_values, "waiting_times": wt_values}, df)
