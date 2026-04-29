from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _build_user_ranges, _prepare_trajectory


def frequency_rank(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the frequency rank of each distinct location for every user.

    The frequency rank ``K_f(r_i)`` of location ``r_i`` is 1 if it is the
    most frequently visited location, 2 if it is the second-most frequently
    visited, and so on.  Ties in visit count are broken by the order they
    appear after sorting (stable), matching the skmob reference implementation.

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
        One row per ``(user, location)`` pair with columns
        ``[uid_col, lat_col, lng_col, "frequency_rank"]``.
        Rank 1 is the most frequently visited location.
        The returned backend matches the input backend.

    @usedBy
        skmob2.measures.visits.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    def _rank_for_values(lat_list: list, lng_list: list) -> tuple[list, list, list[int]]:
        """Compute frequency ranks for a single user's trajectory rows.

        Returns three parallel lists: lats, lngs, ranks — one entry per
        distinct location, ordered by most-frequent-first.
        """
        counts: dict[tuple, int] = {}
        for lat, lng in zip(lat_list, lng_list):
            key = (lat, lng)
            counts[key] = counts.get(key, 0) + 1

        # Sort by count descending (stable: ties keep insertion order).
        sorted_locs = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        lats = [loc[0] for loc, _ in sorted_locs]
        lngs = [loc[1] for loc, _ in sorted_locs]
        ranks = list(range(1, len(lats) + 1))
        return lats, lngs, ranks

    if uid_col is None:
        lats, lngs, ranks = _rank_for_values(df.get_column(lat_col).to_list(), df.get_column(lng_col).to_list())
        return nw.from_dict(
            {lat_col: lats, lng_col: lngs, "frequency_rank": ranks},
            backend=df.implementation,
        ).to_native()

    uid_vals_all: list = []
    lats_all: list = []
    lngs_all: list = []
    ranks_all: list[int] = []

    lat_full = df.get_column(lat_col).to_list()
    lng_full = df.get_column(lng_col).to_list()
    uid_values, ranges = _build_user_ranges(df, uid_col)
    for uid, (start, end) in zip(uid_values, ranges):
        lats, lngs, ranks = _rank_for_values(lat_full[start:end], lng_full[start:end])
        uid_vals_all.extend([uid] * len(lats))
        lats_all.extend(lats)
        lngs_all.extend(lngs)
        ranks_all.extend(ranks)

    return nw.from_dict(
        {
            uid_col: uid_vals_all,
            lat_col: lats_all,
            lng_col: lngs_all,
            "frequency_rank": ranks_all,
        },
        backend=df.implementation,
    ).to_native()
