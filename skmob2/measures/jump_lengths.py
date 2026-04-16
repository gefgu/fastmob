from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import jump_lengths_km

from ._common import _ROW_ORDER_COL, _pick_existing_column


def jump_lengths(traj: Any, show_progress: bool = True, merge: bool = False):
    """Compute jump lengths (km) for each user in the trajectory.

    Parameters
    ----------
    traj
        Trajectory data; must have columns for user ID, datetime, latitude, and longitude.
    show_progress
        Whether to display a progress bar during computation.
    merge
        If True, return a single list of all jump lengths; otherwise, return a dict mapping user IDs to their jump lengths.

    Returns
    -------
    dict[str, list[float]] | list[float]
        Jump lengths in kilometers, either grouped by user or merged into a single list.
    """
    nw_df = nw.from_native(traj, eager_only=True).with_row_index(_ROW_ORDER_COL)

    datetime_col = _pick_existing_column(
        nw_df.columns, ["datetime", "timestamp", "time", "check-in_time"]
    )
    lat_col = _pick_existing_column(nw_df.columns, ["latitude", "lat"])
    lng_col = _pick_existing_column(nw_df.columns, ["longitude", "lon", "lng"])
    uid_col = _pick_existing_column(nw_df.columns, ["user_id", "uid", "user"])

    if not all([datetime_col, lat_col, lng_col]):
        missing = [
            name
            for name, col in zip(
                ["datetime", "latitude", "longitude"],
                [datetime_col, lat_col, lng_col],
            )
            if col is None
        ]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Sort by user and datetime to ensure correct jump length calculations
    sort_cols = (
        [uid_col, datetime_col, _ROW_ORDER_COL]
        if uid_col
        else [datetime_col, _ROW_ORDER_COL]
    )
    df = (
        nw_df.drop_nulls(subset=[datetime_col, lat_col, lng_col])
        .sort(*sort_cols)
        .with_columns(
            nw.col(lat_col).cast(nw.Float64), nw.col(lng_col).cast(nw.Float64)
        )
    )

    lat_buffer = df.get_column(lat_col).to_numpy()
    lon_buffer = df.get_column(lng_col).to_numpy()

    if len(lat_buffer) < 2:
        # Not enough points to compute jump lengths
        return (
            nw.from_dict({uid_col: [], "jump_lengths": []}).to_native()
            if not merge
            else []
        )

    flat_jump_lengths = jump_lengths_km(lat_buffer, lon_buffer)

    # Pad with 0.0 at the start so the array length matches the dataframe.
    # The jump at index `i` becomes the distance traveled from row `i-1` to row `i`.
    padded_jump_lengths = [0.0] + flat_jump_lengths

    if uid_col is None:
        if merge:
            return flat_jump_lengths
        return nw.from_dict(
            {"jump_lengths": [flat_jump_lengths]}, backend=df.implementation
        ).to_native()

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
