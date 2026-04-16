from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import radius_of_gyration_km, radius_of_gyration_batch_km

from ._common import _ROW_ORDER_COL, _pick_existing_column


def radius_of_gyration(traj: Any, show_progress: bool = True):
    """Compute the radius of gyration (km) for each user in the trajectory.

    The radius of gyration captures how far a user typically roams from their
    center of mass.  Formally:

        rg(u) = sqrt( mean_i( haversine(r_i, r_cm)^2 ) )

    where ``r_cm`` is the arithmetic mean of the user's lat/lng coordinates.

    Parameters
    ----------
    traj
        Trajectory data; must have columns for datetime, latitude, and
        longitude.  A user-ID column is optional; when absent the entire
        dataframe is treated as a single individual.
    show_progress
        Accepted for API compatibility with skmob; currently unused.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "radius_of_gyration"]``.
        The returned backend matches the input backend.
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

    lats_full = df.get_column(lat_col).to_list()
    lngs_full = df.get_column(lng_col).to_list()

    if uid_col is None:
        rg = radius_of_gyration_km(list(zip(lats_full, lngs_full)))
        return nw.from_dict(
            {"radius_of_gyration": [rg]}, backend=df.implementation
        ).to_native()

    # Build per-user index ranges from the already-sorted dataframe.
    # The df is sorted by [uid_col, datetime_col] so same-uid rows are contiguous.
    uid_series = df.get_column(uid_col).to_list()
    uid_values: list = []
    ranges: list = []
    i = 0
    n = len(uid_series)
    while i < n:
        current_uid = uid_series[i]
        start = i
        while i < n and uid_series[i] == current_uid:
            i += 1
        uid_values.append(current_uid)
        ranges.append((start, i))

    # Single Rust call covering all users — eliminates per-user boundary crossings
    rog_values = radius_of_gyration_batch_km(lats_full, lngs_full, ranges)

    result = nw.from_dict(
        {uid_col: uid_values, "radius_of_gyration": rog_values},
        backend=df.implementation,
    )

    return result.to_native()
