from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import jump_lengths_km

from .._common import _prepare_trajectory


def jump_lengths(
    traj: Any,
    show_progress: bool = True,
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
    show_progress:
        Accepted for API compatibility with skmob; currently unused.
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

    lat_buffer = df.get_column(lat_col).to_numpy()
    lon_buffer = df.get_column(lng_col).to_numpy()

    if len(lat_buffer) < 2:
        # Not enough points to compute jump lengths
        return nw.from_dict({uid_col: [], "jump_lengths": []}).to_native() if not merge else []

    flat_jump_lengths = jump_lengths_km(lat_buffer, lon_buffer)

    # Pad with 0.0 at the start so the array length matches the dataframe.
    # The jump at index `i` becomes the distance traveled from row `i-1` to row `i`.
    padded_jump_lengths = [0.0] + flat_jump_lengths

    if uid_col is None:
        if merge:
            return flat_jump_lengths
        return nw.from_dict({"jump_lengths": [flat_jump_lengths]}, backend=df.implementation).to_native()

    jumps_df = nw.from_dict(
        {uid_col: df.get_column(uid_col), "jump_lengths": padded_jump_lengths},
        backend=df.implementation,
    )

    # Mask out boundaries (identify where the jump bridged two different users)
    jumps_df = jumps_df.with_columns(nw.col(uid_col).shift(1).alias("__uid_prev"))
    valid_jumps_df = jumps_df.filter(nw.col(uid_col) == nw.col("__uid_prev"))

    if merge:
        return valid_jumps_df.get_column("jump_lengths").to_list()

    jump_dict = {}
    for keys, group in valid_jumps_df.group_by(uid_col):
        jump_dict[keys[0]] = group.get_column("jump_lengths").to_list()

    # Maintain original user order; users with only 1 point get an empty list
    uid_values = df.select(uid_col).unique(maintain_order=True).get_column(uid_col).to_list()
    jump_values = [jump_dict.get(uid, []) for uid in uid_values]

    result = nw.from_dict(
        {uid_col: uid_values, "jump_lengths": jump_values},
        backend=df.implementation,
    )

    return result.to_native()
