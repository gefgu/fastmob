from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _build_user_ranges, _prepare_trajectory


def location_frequency(
    traj: Any,
    normalize: bool = True,
    as_ranks: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return visit frequency for each distinct location per user.

    Counts how many times each ``(lat, lng)`` location was visited by each
    user, optionally normalizing counts to visit probabilities that sum to 1
    within each user.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    normalize:
        When ``True`` (default), the ``"location_frequency"`` column contains
        the probability of visiting that location (count / total_visits), so
        each user's frequencies sum to 1.0.  When ``False``, raw visit counts
        are returned.
    as_ranks:
        When ``True``, return a Python list where element *i* is the mean
        visit frequency of the *i*-th most-visited location across all users
        (rank-1 = most visited).  The list length equals the maximum number
        of distinct locations visited by any single user.  When ``False``
        (default), return a DataFrame.
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
        When ``as_ranks=False``: one row per ``(user, location)`` pair with
        columns ``[uid_col, lat_col, lng_col, "location_frequency"]``.
        When ``as_ranks=True``: a flat Python ``list[float]`` of mean
        per-rank frequencies.
        The returned DataFrame backend matches the input backend.



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
    >>> from skmob2 import location_frequency
    >>> result = location_frequency(df)
    >>> print(result.round({"location_frequency": 3}).head().to_string(index=False))
     uid       lat         lng  location_frequency
       0 39.762146 -104.982480               0.102
       0 39.891077 -105.068532               0.065
       0 39.739154 -104.984703               0.060
       0 39.891586 -105.068463               0.034
       0 39.827022 -105.143191               0.025

    References
    ----------
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

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

    def _freq_for_values(lat_list: list, lng_list: list) -> tuple[list, list, list]:
        counts: dict[tuple, int] = {}
        for lat, lng in zip(lat_list, lng_list):
            key = (lat, lng)
            counts[key] = counts.get(key, 0) + 1

        sorted_locs = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        total = sum(c for _, c in sorted_locs)
        lats = [loc[0] for loc, _ in sorted_locs]
        lngs = [loc[1] for loc, _ in sorted_locs]
        if normalize:
            freqs: list = [c / total for _, c in sorted_locs]
        else:
            freqs = [float(c) for _, c in sorted_locs]

        return lats, lngs, freqs

    if uid_col is None:
        lats, lngs, freqs = _freq_for_values(df.get_column(lat_col).to_list(), df.get_column(lng_col).to_list())
        if as_ranks:
            return freqs
        return nw.from_dict(
            {lat_col: lats, lng_col: lngs, "location_frequency": freqs},
            backend=df.implementation,
        ).to_native()

    uid_vals_all: list = []
    lats_all: list = []
    lngs_all: list = []
    freqs_all: list = []
    per_user_freqs: list[list] = []

    lat_full = df.get_column(lat_col).to_list()
    lng_full = df.get_column(lng_col).to_list()
    uid_values, ranges = _build_user_ranges(df, uid_col)
    for uid, (start, end) in zip(uid_values, ranges):
        lats, lngs, freqs = _freq_for_values(lat_full[start:end], lng_full[start:end])
        uid_vals_all.extend([uid] * len(lats))
        lats_all.extend(lats)
        lngs_all.extend(lngs)
        freqs_all.extend(freqs)
        per_user_freqs.append(freqs)

    if as_ranks:
        max_locs = max(len(f) for f in per_user_freqs) if per_user_freqs else 0
        ranks: list[list] = [[] for _ in range(max_locs)]
        for user_freqs in per_user_freqs:
            for rank_idx, freq in enumerate(user_freqs):
                ranks[rank_idx].append(freq)
        return [sum(r) / len(r) for r in ranks if r]

    return nw.from_dict(
        {
            uid_col: uid_vals_all,
            lat_col: lats_all,
            lng_col: lngs_all,
            "location_frequency": freqs_all,
        },
        backend=df.implementation,
    ).to_native()
