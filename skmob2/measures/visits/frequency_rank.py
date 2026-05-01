from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _build_user_ranges, _is_pandas_backed, _prepare_trajectory


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



    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from skmob2 import frequency_rank
    >>> result = frequency_rank(df)
    >>> print(result.head().to_string(index=False))
     uid       lat         lng  frequency_rank
       0 39.762146 -104.982480               1
       0 39.891077 -105.068532               2
       0 39.739154 -104.984703               3
       0 39.891586 -105.068463               4
       0 39.827022 -105.143191               5

    References
    ----------
    - [BDEM2015] Barbosa, H., de Lima-Neto, F. B., Evsukoff, A., Menezes, R. (2015) The effect of recency to human mobility, EPJ Data Science 4(21), https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-015-0059-8

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

    if _is_pandas_backed(df):
        pd_df = df.to_native()

        def _rank_pandas(frame):
            ranked = (
                frame.groupby([lat_col, lng_col])
                .count()
                .sort_values(by=datetime_col, ascending=False)
                .reset_index()
            )
            ranked["frequency_rank"] = range(1, len(ranked) + 1)
            return ranked[[lat_col, lng_col, "frequency_rank"]]

        if uid_col is None:
            return _rank_pandas(pd_df)

        pieces = []
        for uid, group in pd_df.groupby(uid_col):
            ranked = _rank_pandas(group)
            ranked.insert(0, uid_col, uid)
            pieces.append(ranked)
        if not pieces:
            return pd_df[[uid_col, lat_col, lng_col]].assign(frequency_rank=[]).iloc[0:0]
        import pandas as pd

        return pd.concat(pieces, ignore_index=True)

    def _rank_for_values(lat_list: list, lng_list: list) -> tuple[list, list, list[int]]:
        """Compute frequency ranks for a single user's trajectory rows.

        Returns three parallel lists: lats, lngs, ranks — one entry per
        distinct location, ordered by most-frequent-first.
        """
        counts: dict[tuple, int] = {}
        for lat, lng in zip(lat_list, lng_list):
            key = (lat, lng)
            counts[key] = counts.get(key, 0) + 1

        # Non-pandas backends use a deterministic equivalent for equal-count ties.
        sorted_locs = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))

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
