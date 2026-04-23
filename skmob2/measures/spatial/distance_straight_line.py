from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import total_distance_batch_km

from .._common import _build_user_ranges, _prepare_trajectory


def distance_straight_line(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total trajectory length (km) for each user.

    The total distance (called "distance straight line" in skmob) is the sum
    of Haversine distances between all consecutive GPS fixes in a user's sorted
    trajectory.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
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

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "distance_straight_line"]``.
        Distance values are in kilometres.
        The returned backend matches the input backend.

    @usedBy
        skmob2.measures.spatial.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats_full = df.get_column(lat_col).to_list()
    lngs_full = df.get_column(lng_col).to_list()

    if uid_col is None:
        values = total_distance_batch_km(lats_full, lngs_full, [(0, len(lats_full))])
        return nw.from_dict(
            {"distance_straight_line": values},
            backend=df.implementation,
        ).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    total_distances = total_distance_batch_km(lats_full, lngs_full, ranges)

    return nw.from_dict(
        {uid_col: uid_values, "distance_straight_line": total_distances},
        backend=df.implementation,
    ).to_native()
