from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    jump_lengths_indexed_arrow,
    jump_lengths_indexed_numpy,
    jump_lengths_presorted_arrow,
    jump_lengths_presorted_numpy,
)
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_flat_result_values,
    _build_presorted_user_ends,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_ms,
    _grouped_arrow_values,
    _grouped_numpy_values,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "indexed": jump_lengths_indexed_arrow,
        "presorted": jump_lengths_presorted_arrow,
        "flat_values": _arrow_flat_result_values,
        "group": _grouped_arrow_values,
    },
    numpy_ops={
        "indexed": jump_lengths_indexed_numpy,
        "presorted": jump_lengths_presorted_numpy,
        "flat_values": lambda v: v,
        "group": _grouped_numpy_values,
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
    presorted: bool = False,
):
    """Compute jump lengths (km) for each user in the trajectory.

    A *jump length* is the Haversine distance (in km) between consecutive
    GPS fixes for the same user, sorted by datetime.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge : bool, optional
        When True, return flat jump lengths across all users using a
        backend-appropriate array object.  When False (default), return a
        per-user dataframe.
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
        datetime within each user, then use the presorted contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame or array-like
        When ``merge=False``: a dataframe with columns ``[uid_col, "jump_lengths"]``
        where each row holds an array-like sequence of jump lengths for one
        user.  The returned backend matches the input backend.
        When ``merge=True``: flat jump lengths as a NumPy array for NumPy-backed
        inputs or a PyArrow array for Arrow-backed inputs.


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

    See Also
    --------
    maximum_distance : Largest single jump length per user.
    distance_straight_line : Sum of all jump lengths per user.
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    schema = df.schema
    if schema[lat_col] != nw.Float64 or schema[lng_col] != nw.Float64:
        df = df.with_columns(
            nw.col(lat_col).cast(nw.Float64),
            nw.col(lng_col).cast(nw.Float64),
        )

    ops = _DISPATCHER.get_ops(df)
    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        v_starts, v_ends, flat_values = ops["presorted"](lats_data, lngs_data, ends)
    else:
        timestamps = _extract_timestamps_ms(df, datetime_col)
        timestamps_data = ops["extract_data"](timestamps)
        uid_values, indices, ends = _build_time_ordered_user_ranges(
            df, uid_col, datetime_col, timestamps_data
        )
        v_starts, v_ends, flat_values = ops["indexed"](
            lats_data, lngs_data, indices, ends
        )

    flat_values = ops["flat_values"](flat_values)

    if merge:
        return flat_values

    jump_values = ops["group"](v_starts, v_ends, flat_values, value_offsets=True)
    if uid_col is None:
        return _to_native({"jump_lengths": jump_values}, df)
    return _to_native({uid_col: uid_values, "jump_lengths": jump_values}, df)
