from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    jump_lengths_indexed_arrow,
    jump_lengths_indexed_flat_arrow,
    jump_lengths_indexed_flat_numpy,
    jump_lengths_indexed_numpy,
    jump_lengths_time_ordered_arrow,
    jump_lengths_time_ordered_flat_arrow,
    jump_lengths_time_ordered_flat_numpy,
    jump_lengths_time_ordered_numpy,
    jump_lengths_time_ordered_single_arrow,
    jump_lengths_time_ordered_single_flat_arrow,
    jump_lengths_time_ordered_single_flat_numpy,
    jump_lengths_time_ordered_single_numpy,
)

from .._common import (
    _ROW_ORDER_COL,
    _as_index_array,
    _arrow_result_values,
    _build_user_ranges,
    _extract_timestamps_ms,
    _is_polars_backed,
    _prepare_trajectory,
    _ranges_to_starts_ends,
    _to_native,
    _uid_values_from_index_ranges,
)


def _route_indexed_jump_lengths(
    lats: nw.Series,
    lngs: nw.Series,
    indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    use_arrow: bool,
    merge: bool,
) -> Any:
    if use_arrow:
        if merge:
            return _arrow_flat_result_values(
                jump_lengths_indexed_flat_arrow(lats.to_arrow(), lngs.to_arrow(), indices, starts, ends)
            )
        value_starts, value_ends, values = jump_lengths_indexed_arrow(
            lats.to_arrow(),
            lngs.to_arrow(),
            indices,
            starts,
            ends,
        )
        return _grouped_arrow_values(value_starts, value_ends, values)

    if merge:
        return jump_lengths_indexed_flat_numpy(lats.to_numpy(), lngs.to_numpy(), indices, starts, ends)
    value_starts, value_ends, values = jump_lengths_indexed_numpy(lats.to_numpy(), lngs.to_numpy(), indices, starts, ends)
    return _grouped_numpy_values(value_starts, value_ends, values)


def _arrow_flat_result_values(values: Any) -> Any:
    values = _arrow_result_values(values)
    if hasattr(values, "__arrow_c_array__"):
        import pyarrow as pa

        return pa.array(values)
    return values


def _grouped_numpy_values(value_starts: Any, value_ends: Any, values: Any) -> list[np.ndarray]:
    starts = np.asarray(value_starts, dtype=np.uintp)
    ends = np.asarray(value_ends, dtype=np.uintp)
    values = np.asarray(values, dtype=np.float64)
    return [values[int(start) : int(end)] for start, end in zip(starts, ends)]


def _grouped_arrow_values(value_starts: Any, value_ends: Any, values: Any) -> Any:
    import pyarrow as pa

    starts = np.asarray(value_starts, dtype=np.int64)
    ends = np.asarray(value_ends, dtype=np.int64)
    offsets = np.empty(len(starts) + 1, dtype=np.int32)
    offsets[:-1] = starts
    offsets[-1] = ends[-1] if len(ends) else 0
    return pa.ListArray.from_arrays(offsets, _arrow_flat_result_values(values))


def _route_time_ordered_single_jump_lengths(
    timestamps: nw.Series,
    lats: nw.Series,
    lngs: nw.Series,
    *,
    use_arrow: bool,
    merge: bool,
) -> Any:
    if use_arrow:
        if merge:
            return _arrow_flat_result_values(
                jump_lengths_time_ordered_single_flat_arrow(
                    timestamps.to_arrow(),
                    lats.to_arrow(),
                    lngs.to_arrow(),
                )
            )
        value_starts, value_ends, values = jump_lengths_time_ordered_single_arrow(
            timestamps.to_arrow(),
            lats.to_arrow(),
            lngs.to_arrow(),
        )
        return _grouped_arrow_values(value_starts, value_ends, values)

    if merge:
        return jump_lengths_time_ordered_single_flat_numpy(timestamps.to_numpy(), lats.to_numpy(), lngs.to_numpy())
    value_starts, value_ends, values = jump_lengths_time_ordered_single_numpy(
        timestamps.to_numpy(),
        lats.to_numpy(),
        lngs.to_numpy(),
    )
    return _grouped_numpy_values(value_starts, value_ends, values)


def _route_time_ordered_jump_lengths(
    uids: nw.Series,
    timestamps: nw.Series,
    lats: nw.Series,
    lngs: nw.Series,
    *,
    use_arrow: bool,
    merge: bool,
) -> tuple[Any, Any, Any, Any]:
    if use_arrow:
        if merge:
            indices, starts, ends, values = jump_lengths_time_ordered_flat_arrow(
                uids.to_arrow(),
                timestamps.to_arrow(),
                lats.to_arrow(),
                lngs.to_arrow(),
            )
            return indices, starts, ends, _arrow_flat_result_values(values)
        indices, starts, ends, value_starts, value_ends, values = jump_lengths_time_ordered_arrow(
            uids.to_arrow(),
            timestamps.to_arrow(),
            lats.to_arrow(),
            lngs.to_arrow(),
        )
        return indices, starts, ends, _grouped_arrow_values(value_starts, value_ends, values)

    if merge:
        return jump_lengths_time_ordered_flat_numpy(uids.to_numpy(), timestamps.to_numpy(), lats.to_numpy(), lngs.to_numpy())
    indices, starts, ends, value_starts, value_ends, values = jump_lengths_time_ordered_numpy(
        uids.to_numpy(),
        timestamps.to_numpy(),
        lats.to_numpy(),
        lngs.to_numpy(),
    )
    return indices, starts, ends, _grouped_numpy_values(value_starts, value_ends, values)


def _build_time_ordered_ranges_fallback(
    df: nw.DataFrame,
    uid_col: str,
    datetime_col: str,
) -> tuple[list, list[int], list[tuple[int, int]]]:
    index_df = (
        df.select([uid_col, datetime_col])
        .with_row_index(_ROW_ORDER_COL)
        .sort(uid_col, datetime_col, _ROW_ORDER_COL)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(_ROW_ORDER_COL).to_list()]
    return uid_values, indices, ranges


def jump_lengths(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
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
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
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

    if uid_col is None:
        jump_values = _route_time_ordered_single_jump_lengths(
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
        indices, starts, ends, jump_values = _route_time_ordered_jump_lengths(
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
        starts, ends = _ranges_to_starts_ends(ranges)
        indices = _as_index_array(indices)
        jump_values = _route_indexed_jump_lengths(
            lats_full,
            lngs_full,
            indices,
            starts,
            ends,
            use_arrow=use_arrow,
            merge=merge,
        )

    if merge:
        return jump_values

    return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)
