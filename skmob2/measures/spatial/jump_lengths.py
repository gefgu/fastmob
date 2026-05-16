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
from .._common import (
    _ROW_ORDER_COL,
    _as_index_array,
    _arrow_result_values,
    _build_user_ranges,
    _extract_timestamps_ms,
    _is_polars_backed,
    _detect_trajectory_columns,
    _prepare_trajectory,
    _to_native,
    _uid_values_from_index_ranges,
)


def _route_presorted_jump_lengths(
    lats: Any,
    lngs: Any,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
    merge: bool,
) -> Any:
    if use_arrow:
        starts, ends, values = jump_lengths_presorted_arrow(
            lats.to_arrow() if hasattr(lats, "to_arrow") else lats,
            lngs.to_arrow() if hasattr(lngs, "to_arrow") else lngs,
            ranges,
        )
        values = _arrow_flat_result_values(values)
        if merge:
            return values
        return _grouped_arrow_values(starts, ends, values, value_offsets=True)

    starts, ends, values = jump_lengths_presorted_numpy(
        lats.to_numpy() if hasattr(lats, "to_numpy") else lats,
        lngs.to_numpy() if hasattr(lngs, "to_numpy") else lngs,
        ranges,
    )
    if merge:
        return values
    return _grouped_numpy_values(starts, ends, values, value_offsets=True)


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


def _route_non_ordered_jump_lengths(
    uids: nw.Series | None,
    timestamps: nw.Series,
    lats: nw.Series,
    lngs: nw.Series,
    *,
    use_arrow: bool,
    merge: bool,
) -> tuple[Any, Any, Any, Any]:
    if use_arrow:
        indices, starts, ends, values = jump_lengths_non_ordered_arrow(
            None if uids is None else uids.to_arrow(),
            timestamps.to_arrow(),
            lats.to_arrow(),
            lngs.to_arrow(),
        )
        values = _arrow_flat_result_values(values)
        if merge:
            return indices, starts, ends, values
        return (
            indices,
            starts,
            ends,
            _grouped_arrow_values(starts, ends, values),
        )

    indices, starts, ends, values = jump_lengths_non_ordered_numpy(
        None if uids is None else uids.to_numpy(),
        timestamps.to_numpy(),
        lats.to_numpy(),
        lngs.to_numpy(),
    )
    if merge:
        return indices, starts, ends, values
    return (
        indices,
        starts,
        ends,
        _grouped_numpy_values(starts, ends, values),
    )


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
    if uid_col is None:
        return None, [(0, len(df))]

    uid_values, ranges = _build_user_ranges(df, uid_col)
    return uid_values, ranges


def _presorted_coordinate_series_from_indices(
    lats: nw.Series,
    lngs: nw.Series,
    indices: Any,
    *,
    use_arrow: bool,
) -> tuple[Any, Any]:
    indices = np.asarray(indices, dtype=np.uintp)
    if use_arrow:
        import pyarrow as pa

        take_indices = pa.array(indices.astype(np.int64, copy=False))
        return lats.to_arrow().take(take_indices), lngs.to_arrow().take(take_indices)

    return lats.to_numpy()[indices], lngs.to_numpy()[indices]


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

    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    timestamps = _extract_timestamps_ms(df, datetime_col)
    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if sorted:
        uid_values, ranges = _presorted_ranges(df, uid_col)
        jump_values = _route_presorted_jump_lengths(
            lats_full,
            lngs_full,
            ranges,
            use_arrow=use_arrow,
            merge=merge,
        )
        if merge:
            return jump_values
        if uid_col is None:
            return _to_native({"jump_lengths": jump_values}, df)
        return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)

    if uid_col is None:
        _indices, _starts, _ends, jump_values = _route_non_ordered_jump_lengths(
            None,
            timestamps,
            lats_full,
            lngs_full,
            use_arrow=use_arrow,
            merge=merge,
        )
        if merge:
            return jump_values
        return _to_native({"jump_lengths": jump_values}, df)

    uids = df.get_column(uid_col)
    try:
        indices, starts, ends, jump_values = _route_non_ordered_jump_lengths(
            uids,
            timestamps,
            lats_full,
            lngs_full,
            use_arrow=use_arrow,
            merge=merge,
        )
        indices = _as_index_array(indices)
        starts = _as_index_array(starts)
        ends = _as_index_array(ends)
        uid_values = _uid_values_from_index_ranges(uids, indices, starts, use_arrow=use_arrow)
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise
        uid_values, indices, ranges = _build_time_ordered_ranges_fallback(df, uid_col, datetime_col)
        sorted_lats, sorted_lngs = _presorted_coordinate_series_from_indices(
            lats_full, lngs_full, indices, use_arrow=use_arrow
        )
        jump_values = _route_presorted_jump_lengths(
            sorted_lats,
            sorted_lngs,
            ranges,
            use_arrow=use_arrow,
            merge=merge,
        )

    if merge:
        return jump_values

    return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)
