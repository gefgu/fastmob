from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob._core import interpolate_at_indexed, interpolate_at_presorted
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _arrow_flat_result_values,
    _build_presorted_user_ends,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _take_uid_values,
    _timestamps_s_to_datetime_ns,
    _to_native,
)

_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})

_INTERPOLATE_AT_METHODS = ("linear", "nearest")


def _query_timestamps_s(at: Any) -> np.ndarray:
    """Normalize a scalar or sequence of timestamp-likes into a float64 seconds-since-epoch array.

    Accepts anything NumPy's ``datetime64`` constructor understands (Python
    ``datetime``, ``numpy.datetime64``, ISO-8601 strings, pandas
    ``Timestamp`` objects, ...) without importing pandas/polars directly
    (forbidden under `fastmob/`, see CLAUDE.md).
    """
    values = at if isinstance(at, (list, tuple, np.ndarray)) else [at]
    arr = np.asarray(values, dtype="datetime64[ns]")
    return arr.astype("int64").astype("float64") / 1e9


def interpolate_at(
    traj: Any,
    at: Any,
    method: str = "linear",
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Query each user's interpolated position at one or more timestamps.

    Unlike :func:`fastmob.trajectory.interpolate`, this never changes a
    user's own point count -- it answers "where was this user at time t?"
    independently per user and per query timestamp. A query time outside a
    user's own ``[min(datetime), max(datetime)]`` range is marked invalid
    (``valid=False``, ``NaN`` position) rather than raising, matching this
    codebase's validity-mask convention over exceptions.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    at:
        A single timestamp-like, or a sequence of timestamp-likes. Every
        user is queried at every timestamp in `at`.
    method:
        ``"linear"`` (default) interpolates position between the two
        surrounding points; ``"nearest"`` returns the closer of the two.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    presorted:
        Whether the trajectory is already sorted by user and time.

    Returns
    -------
    DataFrame
        One row per ``(uid, query_time)`` pair (or one row per query
        timestamp when no user column is present), with columns ``uid``
        (when present), ``query_time``, `lat_col`, `lng_col`, and ``valid``.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> df = pd.DataFrame({
    ...     "uid": [1, 1, 1],
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ...     "datetime": pd.to_datetime(
    ...         ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    ...     ),
    ... })
    >>> from fastmob.trajectory import interpolate_at
    >>> out = interpolate_at(df, at="2020-01-01 00:30", method="linear")
    >>> bool(out["valid"].iloc[0])
    True
    """
    if method not in _INTERPOLATE_AT_METHODS:
        raise ValueError(f"unknown interpolate_at method: {method!r}; choose from {sorted(_INTERPOLATE_AT_METHODS)}")

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

    ops = _EXTRACTOR.get_ops(df)
    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    timestamps_s = _extract_timestamps_s(df, datetime_col)
    times_data = ops["extract_data"](timestamps_s)

    query_times_s = _query_timestamps_s(at)

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        out_lats, out_lngs, out_valid = interpolate_at_presorted(
            lats_data, lngs_data, times_data, ends, query_times_s, method
        )
    else:
        uid_values, indices, ends = _build_time_ordered_user_ranges(df, uid_col, datetime_col, times_data)
        out_lats, out_lngs, out_valid = interpolate_at_indexed(
            lats_data, lngs_data, times_data, indices, ends, query_times_s, method
        )

    num_users = 1 if uid_values is None else len(uid_values)
    num_queries = len(query_times_s)
    user_positions = np.repeat(np.arange(num_users), num_queries)
    query_time_values = np.tile(query_times_s, num_users)

    result_dict: dict[str, Any] = {}
    if uid_col is not None:
        result_dict[uid_col] = _take_uid_values(uid_values, user_positions)
    result_dict["query_time"] = _timestamps_s_to_datetime_ns(query_time_values)
    result_dict[lat_col] = _arrow_flat_result_values(out_lats)
    result_dict[lng_col] = _arrow_flat_result_values(out_lngs)
    result_dict["valid"] = _arrow_flat_result_values(out_valid)

    return _to_native(result_dict, df)


interpolate_at.__module__ = "fastmob.trajectory"
