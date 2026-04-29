from __future__ import annotations

from typing import Any, Literal

import narwhals as nw
from skmob2._core import (
    radius_of_gyration_arrow,
    radius_of_gyration_indexed_arrow,
    radius_of_gyration_indexed_numpy,
    radius_of_gyration_numpy,
    radius_of_gyration_user_indices_arrow,
    radius_of_gyration_user_indices_numpy,
)

from .._common import _build_user_ranges, _prepare_trajectory

SortStrategy = Literal["sort", "indexed", "presorted"]
_SORT_STRATEGIES = {"sort", "indexed", "presorted"}
_ROG_ROW_INDEX_COL = "__skmob2_rog_row_index__"


def _is_polars_backed(nw_df: nw.DataFrame) -> bool:
    native = nw_df.to_native()
    return hasattr(native, "lazy")


def _route_and_call(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return radius_of_gyration_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    return radius_of_gyration_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def _route_and_call_indexed(
    lats: nw.Series,
    lngs: nw.Series,
    indices: list[int],
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return radius_of_gyration_indexed_arrow(lats.to_arrow(), lngs.to_arrow(), indices, ranges)

    return radius_of_gyration_indexed_numpy(lats.to_numpy(), lngs.to_numpy(), indices, ranges)


def _validate_sort_strategy(sort_strategy: str) -> SortStrategy:
    if sort_strategy not in _SORT_STRATEGIES:
        raise ValueError(
            "sort_strategy must be one of 'sort', 'indexed', or 'presorted'; "
            f"got {sort_strategy!r}"
        )
    return sort_strategy  # type: ignore[return-value]


def _build_indexed_user_ranges(df: nw.DataFrame, uid_col: str) -> tuple[list, list[int], list[tuple[int, int]]]:
    n = len(df)
    if n == 0:
        return [], [], []

    uid_series = df.get_column(uid_col)
    try:
        if _is_polars_backed(df):
            # 1. Call to_arrow exactly once
            uid_arrow = uid_series.to_arrow()
            indices, ranges = radius_of_gyration_user_indices_arrow(uid_arrow)
            
            # 2. Skip .take() and Arrow's compute overhead by using Python list comprehension directly 
            # on the Arrow array, which supports __getitem__ indexing.
            uid_values = [uid_arrow[indices[start]].as_py() for start, _ in ranges]
        else:
            indices, ranges = radius_of_gyration_user_indices_numpy(uid_series.to_numpy())
            start_indices = [indices[start] for start, _ in ranges]
            uid_values = uid_series.to_numpy()[start_indices].tolist()
        return uid_values, indices, ranges
    except ValueError:
        pass

    index_df = (
        df.select([uid_col])
        .with_row_index(_ROG_ROW_INDEX_COL)
        .sort(uid_col, _ROG_ROW_INDEX_COL)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(_ROG_ROW_INDEX_COL).to_list()]
    return uid_values, indices, ranges


def radius_of_gyration(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sort_strategy: SortStrategy = "sort",
):
    """Compute the radius of gyration (km) for each user in the trajectory.

    The radius of gyration captures how far a user typically roams from their
    center of mass.  Formally::

        rg(u) = sqrt( mean_i( haversine(r_i, r_cm)^2 ) )

    where ``r_cm`` is the arithmetic mean of the user's lat/lng coordinates.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
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
    sort_strategy:
        Strategy used to group trajectory rows by user:

        - ``"sort"`` sorts the dataframe by user and datetime first. This is
          the fastest general-purpose path, but has the highest peak memory
          usage.
        - ``"indexed"`` keeps the dataframe in input order and sorts only a
          skinny user/index table. It uses less peak memory, but can be slower
          because coordinates are read through scattered indexes.
        - ``"presorted"`` keeps the dataframe in input order and assumes rows
          are already grouped contiguously by user. This is a fast low-memory
          path, but wrong grouping can produce wrong or duplicate user rows.

        Radius of gyration is order-independent, so chronological sorting is
        not required for correctness; grouping rows by user is what matters.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "radius_of_gyration"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures import radius_of_gyration
    >>> df = pd.DataFrame({
    ...     "uid": ["a", "a", "a"],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ... })
    >>> radius_of_gyration(df)
    """
    sort_strategy = _validate_sort_strategy(sort_strategy)
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=sort_strategy == "sort",
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        (rg,) = _route_and_call(lats_full, lngs_full, [(0, len(df))], use_arrow=use_arrow)
        return nw.from_dict({"radius_of_gyration": [rg]}, backend=df.implementation).to_native()

    if sort_strategy == "indexed":
        uid_values, indices, ranges = _build_indexed_user_ranges(df, uid_col)
        rog_values = _route_and_call_indexed(lats_full, lngs_full, indices, ranges, use_arrow=use_arrow)
    else:
        uid_values, ranges = _build_user_ranges(df, uid_col)
        # Single Rust call covering all users eliminates per-user boundary crossings.
        rog_values = _route_and_call(lats_full, lngs_full, ranges, use_arrow=use_arrow)

    result = nw.from_dict(
        {uid_col: uid_values, "radius_of_gyration": rog_values},
        backend=df.implementation,
    )

    return result.to_native()
