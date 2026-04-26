from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def number_of_visits(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total number of trajectory points (visits) for each user.

    A "visit" is defined as one row in the trajectory dataframe after null
    removal. The result is the row count per user — identical to the skmob
    ``number_of_visits`` individual measure.

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
        One row per user with columns ``[uid_col, "number_of_visits"]``.
        The returned backend matches the input backend.

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    if uid_col is None:
        count = len(df)
        return nw.from_dict(
            {"number_of_visits": [count]},
            backend=df.implementation,
        ).to_native()

    result = df.group_by(uid_col).agg(nw.len().alias("number_of_visits")).sort(uid_col)

    return result.to_native()
