from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import waiting_times_seconds as _waiting_times_seconds_rust

from .._common import _build_user_ranges, _prepare_trajectory


def waiting_times(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the waiting times (seconds) between consecutive GPS fixes for each user.

    The waiting time at position i is the number of seconds elapsed between
    trajectory point i and point i+1, within the same user's sorted trajectory.
    Users with fewer than 2 points receive an empty list.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge:
        When ``True``, return a single flat Python list of all waiting times
        across all users concatenated together.  When ``False`` (default),
        return a DataFrame with one row per user.
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
    DataFrame or list
        When ``merge=False``: one row per user with columns
        ``[uid_col, "waiting_times"]``; each cell is a list of floats (seconds).
        When ``merge=True``: a flat Python ``list[float]`` of all waiting times.
        The returned DataFrame backend matches the input backend.

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

    # Extract Unix timestamps in seconds via millisecond intermediate to avoid
    # backend-specific nanosecond vs microsecond differences.
    timestamps_s: list[float] = (
        df.with_columns((nw.col(datetime_col).dt.timestamp("ms") / 1000.0).alias("__ts_s__"))
        .get_column("__ts_s__")
        .to_list()
    )

    if uid_col is None:
        wt_list = _waiting_times_seconds_rust(timestamps_s, [(0, len(timestamps_s))])
        if merge:
            return wt_list[0]
        return nw.from_dict(
            {"waiting_times": wt_list},
            backend=df.implementation,
        ).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    wt_lists = _waiting_times_seconds_rust(timestamps_s, ranges)

    if merge:
        flat: list = []
        for wt in wt_lists:
            flat.extend(wt)
        return flat

    return nw.from_dict(
        {uid_col: uid_values, "waiting_times": wt_lists},
        backend=df.implementation,
    ).to_native()
