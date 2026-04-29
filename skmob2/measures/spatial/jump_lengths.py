from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    jump_lengths_arrow,
    jump_lengths_flat_arrow,
    jump_lengths_flat_numpy,
    jump_lengths_numpy,
)

from .._common import _build_user_ranges, _is_polars_backed, _prepare_trajectory


def _route_jump_lengths(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
    merge: bool,
) -> list[float] | list[list[float]]:
    if use_arrow:
        if merge:
            return jump_lengths_flat_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)
        return jump_lengths_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    if merge:
        return jump_lengths_flat_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)
    return jump_lengths_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


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
        When True, return a flat ``list[float]`` of all jump lengths across
        all users.  When False (default), return a per-user dataframe.
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
    DataFrame | list[float]
        When ``merge=False``: a dataframe with columns ``[uid_col, "jump_lengths"]``
        where each row holds a list of jump lengths for one user.  The returned
        backend matches the input backend.
        When ``merge=True``: a flat ``list[float]`` of all jump lengths.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures import jump_lengths
    >>> df = pd.DataFrame({
    ...     "uid": ["a", "a", "a"],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ... })
    >>> jump_lengths(df)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        ranges = [(0, len(df))]
        jump_values = _route_jump_lengths(lats_full, lngs_full, ranges, use_arrow=use_arrow, merge=merge)
        if merge:
            return jump_values
        return nw.from_dict({"jump_lengths": jump_values}, backend=df.implementation).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    jump_values = _route_jump_lengths(lats_full, lngs_full, ranges, use_arrow=use_arrow, merge=merge)
    if merge:
        return jump_values

    result = nw.from_dict(
        {uid_col: uid_values, "jump_lengths": jump_values},
        backend=df.implementation,
    )

    return result.to_native()
