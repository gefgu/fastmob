from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    jump_lengths_non_ordered_arrow,
    jump_lengths_non_ordered_numpy,
    jump_lengths_presorted_arrow,
    jump_lengths_presorted_numpy,
)
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _ROW_ORDER_COL,
    _as_index_array,
    _arrow_result_values,
    _build_presorted_user_ranges,
    _build_user_ranges,
    _extract_timestamps_ms,
    _detect_trajectory_columns,
    _to_native,
    _uid_values_from_index_ranges,
)


def _route_presorted_jump_lengths(
    ops: dict,
    lats_data: Any,
    lngs_data: Any,
    ranges: list[tuple[int, int]],
    *,
    merge: bool,
) -> Any:
    starts, ends, values = ops["presorted"](lats_data, lngs_data, ranges)
    values = ops["flat_values"](values)
    if merge:
        return values
    return ops["group"](starts, ends, values, value_offsets=True)


def _arrow_flat_result_values(values: Any) -> Any:
    values = _arrow_result_values(values)
    if hasattr(values, "__arrow_c_array__"):
        import pyarrow as pa

        return pa.array(values)
    return values


def _grouped_numpy_values(starts: Any, ends: Any, values: Any, *, value_offsets: bool = False) -> list[np.ndarray]:
    if value_offsets:
        starts = np.asarray(starts, dtype=np.uintp)
        ends = np.asarray(ends, dtype=np.uintp)
    else:
        starts, ends = _value_offsets_from_index_ranges(starts, ends)
    values = np.asarray(values, dtype=np.float64)
    return [values[int(start) : int(end)] for start, end in zip(starts, ends)]


def _grouped_arrow_values(starts: Any, ends: Any, values: Any, *, value_offsets: bool = False) -> Any:
    import pyarrow as pa

    if value_offsets:
        starts = np.asarray(starts, dtype=np.uintp)
        ends = np.asarray(ends, dtype=np.uintp)
    else:
        starts, ends = _value_offsets_from_index_ranges(starts, ends)
    offsets = np.empty(len(starts) + 1, dtype=np.int32)
    offsets[:-1] = starts
    offsets[-1] = ends[-1] if len(ends) else 0
    return pa.ListArray.from_arrays(offsets, _arrow_flat_result_values(values))


def _value_offsets_from_index_ranges(index_starts: Any, index_ends: Any) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(index_starts, dtype=np.uintp)
    ends = np.asarray(index_ends, dtype=np.uintp)
    lengths = np.maximum(ends - starts, 1) - 1
    value_ends = np.cumsum(lengths, dtype=np.uintp)
    value_starts = value_ends - lengths
    return value_starts, value_ends


def _take_arrow_coords(lats: nw.Series, lngs: nw.Series, indices: np.ndarray) -> tuple[Any, Any]:
    import pyarrow as pa

    take_idx = pa.array(indices.astype(np.int64, copy=False))
    return lats.to_arrow().take(take_idx), lngs.to_arrow().take(take_idx)


def _take_numpy_coords(lats: nw.Series, lngs: nw.Series, indices: np.ndarray) -> tuple[Any, Any]:
    arr = lats.to_numpy()
    return arr[indices], lngs.to_numpy()[indices]


def _route_non_ordered_jump_lengths(
    ops: dict,
    uids_data: Any,
    timestamps_data: Any,
    lats_data: Any,
    lngs_data: Any,
    *,
    merge: bool,
) -> tuple[Any, Any, Any, Any]:
    indices, starts, ends, values = ops["non_ordered"](uids_data, timestamps_data, lats_data, lngs_data)
    values = ops["flat_values"](values)
    if merge:
        return indices, starts, ends, values
    return (indices, starts, ends, ops["group"](starts, ends, values))


def _build_time_ordered_ranges_fallback(
    df: nw.DataFrame,
    uid_col: str,
    datetime_col: str,
) -> tuple[list, list[int], list[tuple[int, int]]]:
    index_df = (
        df.select([uid_col, datetime_col]).with_row_index(_ROW_ORDER_COL).sort(uid_col, datetime_col, _ROW_ORDER_COL)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(_ROW_ORDER_COL).to_list()]
    return uid_values, indices, ranges


def _presorted_ranges(df: nw.DataFrame, uid_col: str | None) -> tuple[list | None, list[tuple[int, int]]]:
    return _build_presorted_user_ranges(df, uid_col)


def _presorted_coordinate_series_from_indices(
    ops: dict,
    lats: nw.Series,
    lngs: nw.Series,
    indices: Any,
) -> tuple[Any, Any]:
    indices = np.asarray(indices, dtype=np.uintp)
    return ops["take_coords"](lats, lngs, indices)


_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "non_ordered": jump_lengths_non_ordered_arrow,
        "presorted": jump_lengths_presorted_arrow,
        "flat_values": _arrow_flat_result_values,
        "group": _grouped_arrow_values,
        "take_coords": _take_arrow_coords,
    },
    numpy_ops={
        "non_ordered": jump_lengths_non_ordered_numpy,
        "presorted": jump_lengths_presorted_numpy,
        "flat_values": lambda v: v,
        "group": _grouped_numpy_values,
        "take_coords": _take_numpy_coords,
    },
)


def jump_lengths(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
):
    """Compute jump lengths (km) for each user in the trajectory.

    A *jump length* is the Haversine distance (in km) between consecutive
    GPS fixes for the same user, sorted by datetime.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge:
        When True, return flat jump lengths across all users using a
        backend-appropriate array object.  When False (default), return a
        per-user dataframe.
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
        datetime within each user, then use the presorted contiguous fast path.

    Returns
    -------
    DataFrame | array-like
        When ``merge=False``: a dataframe with columns ``[uid_col, "jump_lengths"]``
        where each row holds an array-like sequence of jump lengths for one
        user.  The returned backend matches the input backend.
        When ``merge=True``: flat jump lengths as a NumPy array for NumPy-backed
        inputs or a PyArrow array for Arrow-backed inputs. Unsupported fallback
        paths may return a Python list.



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
    >>> from skmob2 import jump_lengths
    >>> result = jump_lengths(df)
    >>> preview = result.assign(n_jumps=result["jump_lengths"].str.len())
    >>> print(preview[["uid", "n_jumps"]].head().to_string(index=False))
     uid  n_jumps
       0     2098
       1     1209
       2     1690

    References
    ----------
    - [BHG2006] Brockmann, D., Hufnagel, L. & Geisel, T. (2006) The scaling laws of human travel. Nature 439, 462-465, https://www.nature.com/articles/nature04292
    - [GHB2008] Gonzalez, M. C., Hidalgo, C. A. & Barabasi, A. L. (2008) Understanding individual human mobility patterns. Nature, 453, 779-782, https://www.nature.com/articles/nature06958.
    - [PRQPG2013] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D. & Giannotti, F. (2013) Understanding the patterns of car travel. European Physics Journal Special Topics 215(1), 61-73, https://link.springer.com/article/10.1140%2Fepjst%2Fe2013-01715-5


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
    timestamps = _extract_timestamps_ms(df, datetime_col)
    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    timestamps_data = ops["extract_data"](timestamps)
    lats_data = ops["extract_data"](lats_full)
    lngs_data = ops["extract_data"](lngs_full)

    if sorted:
        uid_values, ranges = _presorted_ranges(df, uid_col)
        jump_values = _route_presorted_jump_lengths(ops, lats_data, lngs_data, ranges, merge=merge)
        if merge:
            return jump_values
        if uid_col is None:
            return _to_native({"jump_lengths": jump_values}, df)
        return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)

    if uid_col is None:
        _indices, _starts, _ends, jump_values = _route_non_ordered_jump_lengths(
            ops, None, timestamps_data, lats_data, lngs_data, merge=merge
        )
        if merge:
            return jump_values
        return _to_native({"jump_lengths": jump_values}, df)

    uids = df.get_column(uid_col)
    uids_data = ops["extract_data"](uids)
    try:
        indices, starts, ends, jump_values = _route_non_ordered_jump_lengths(
            ops, uids_data, timestamps_data, lats_data, lngs_data, merge=merge
        )
        indices = _as_index_array(indices)
        starts = _as_index_array(starts)
        ends = _as_index_array(ends)
        uid_values = _uid_values_from_index_ranges(uids, indices, ends, use_arrow=use_arrow)
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise
        uid_values, indices, ranges = _build_time_ordered_ranges_fallback(df, uid_col, datetime_col)
        sorted_lats, sorted_lngs = _presorted_coordinate_series_from_indices(
            ops, lats_full, lngs_full, indices
        )
        jump_values = _route_presorted_jump_lengths(ops, sorted_lats, sorted_lngs, ranges, merge=merge)

    if merge:
        return jump_values

    return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)
