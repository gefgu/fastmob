from __future__ import annotations

from datetime import timedelta
from typing import Any

from .._common import _prepare_trajectory
from ..._core import square_displacement_km2


def mean_square_displacement(
    traj: Any,
    *,
    days: int = 0,
    hours: int = 1,
    minutes: int = 0,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> float:
    """Return the mean square displacement (km²) across all users.

    For each user the square displacement is computed as the squared Haversine
    distance (in km) between their first recorded position ``r0`` and the last
    position whose timestamp satisfies ``datetime <= r0_time + delta_t``, where
    ``delta_t = timedelta(days=days, hours=hours, minutes=minutes)``.

    The function then returns the arithmetic mean of these per-user values —
    matching the skmob ``collective.mean_square_displacement`` formula exactly.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.  The
        trajectory must be sorted by datetime within each user (``_prepare_trajectory``
        performs this sort automatically).
    days:
        Days component of the time offset from each user's start time.
        Defaults to 0.
    hours:
        Hours component of the time offset.  Defaults to 1.
    minutes:
        Minutes component of the time offset.  Defaults to 0.
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
    float
        Mean square displacement in km².  Returns 0.0 when the trajectory is
        empty or every user's displacement window is trivially at the start.

    @usedBy
        skmob2.measures.flows.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    delta_t = timedelta(days=days, hours=hours, minutes=minutes)

    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    # _prepare_trajectory already sorts by [uid, datetime], so within each user
    # rows are in ascending time order.
    records = df.to_native()

    # Identify the uid groups. When there is no uid column _prepare_trajectory
    # returns uid_col=None; we treat the whole frame as one group.
    if uid_col is None:
        groups = {"__all__": records}
    else:
        groups = {uid: grp for uid, grp in records.groupby(uid_col, sort=False)}

    sq_displacements: list[float] = []
    for uid, grp in groups.items():
        # Ensure rows are time-sorted (groupby in pandas does not guarantee order).
        grp = grp.sort_values(datetime_col)

        r0 = grp.iloc[0]
        t_limit = r0[datetime_col] + delta_t

        # Last point within the time window.
        window = grp[grp[datetime_col] <= t_limit]
        if window.empty:
            # No rows satisfy the constraint; use r0 as rt (displacement = 0).
            rt = r0
        else:
            rt = window.iloc[-1]

        d2 = square_displacement_km2(
            float(r0[lat_col]),
            float(r0[lng_col]),
            float(rt[lat_col]),
            float(rt[lng_col]),
        )
        sq_displacements.append(d2)

    if not sq_displacements:
        return 0.0

    return sum(sq_displacements) / len(sq_displacements)
