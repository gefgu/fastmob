from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import maximum_distance_arrow, maximum_distance_numpy

from .._common import _build_user_ranges, _is_polars_backed, _prepare_trajectory


def _route_maximum_distance(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return maximum_distance_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)
    return maximum_distance_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def maximum_distance(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the maximum distance (km) covered in a single movement for each user.

    The maximum distance is the largest Haversine distance between any two
    consecutive GPS fixes in a user's sorted trajectory.

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
        One row per user with columns ``[uid_col, "maximum_distance"]``.
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

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        values = _route_maximum_distance(lats_full, lngs_full, [(0, len(df))], use_arrow=use_arrow)
        return nw.from_dict(
            {"maximum_distance": values},
            backend=df.implementation,
        ).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    max_distances = _route_maximum_distance(lats_full, lngs_full, ranges, use_arrow=use_arrow)

    return nw.from_dict(
        {uid_col: uid_values, "maximum_distance": max_distances},
        backend=df.implementation,
    ).to_native()
