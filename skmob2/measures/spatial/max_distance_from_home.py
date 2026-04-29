from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import max_distance_from_point_batch_km

from .._common import _build_user_ranges, _prepare_trajectory
from .home_location import home_location


def max_distance_from_home(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the maximum Haversine distance (km) from each user's home location.

    The home location is computed via :func:`home_location` using the
    nighttime-window heuristic.  The distance is the maximum Haversine
    distance from that home point to any trajectory point for the user.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    start_night:
        Hour (0–23) at which the nighttime window begins.  Forwarded to
        :func:`home_location`.  Default: 22.
    end_night:
        Hour (0–23) at which the nighttime window ends (exclusive).  Forwarded
        to :func:`home_location`.  Default: 7.
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
        One row per user with columns ``[uid_col, "max_distance_from_home"]``.
        Distance values are in kilometres.  The returned backend matches the
        input backend.

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

    # Compute home locations using the same column names.
    home_df = nw.from_native(
        home_location(
            traj,
            start_night=start_night,
            end_night=end_night,
            datetime_col=datetime_col,
            lat_col=lat_col,
            lng_col=lng_col,
            uid_col=uid_col,
        ),
        eager_only=True,
    )

    lats_full = df.get_column(lat_col).to_list()
    lngs_full = df.get_column(lng_col).to_list()

    if uid_col is None:
        home_lat = home_df.get_column(lat_col).to_list()[0]
        home_lng = home_df.get_column(lng_col).to_list()[0]
        values = max_distance_from_point_batch_km([home_lat], [home_lng], lats_full, lngs_full, [(0, len(lats_full))])
        return nw.from_dict(
            {"max_distance_from_home": values},
            backend=df.implementation,
        ).to_native()

    # Build a dict of uid -> (home_lat, home_lng) from the home_location result.
    home_map: dict = {}
    for row in home_df.rows(named=True):
        home_map[row[uid_col]] = (row[lat_col], row[lng_col])

    uid_values_all, ranges_all = _build_user_ranges(df, uid_col)
    uid_values: list = []
    ranges: list[tuple[int, int]] = []
    home_lats: list[float] = []
    home_lngs: list[float] = []

    for current_uid, user_range in zip(uid_values_all, ranges_all):
        if current_uid in home_map:
            uid_values.append(current_uid)
            ranges.append(user_range)
            h_lat, h_lng = home_map[current_uid]
            home_lats.append(h_lat)
            home_lngs.append(h_lng)

    max_distances = max_distance_from_point_batch_km(home_lats, home_lngs, lats_full, lngs_full, ranges)

    return nw.from_dict(
        {uid_col: uid_values, "max_distance_from_home": max_distances},
        backend=df.implementation,
    ).to_native()
