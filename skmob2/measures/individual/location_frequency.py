from __future__ import annotations
import narwhals as nw

from typing import Any

from skmob2._core import location_frequency_indexed_arrow, location_frequency_indexed_numpy
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _detect_trajectory_columns,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "kernel": location_frequency_indexed_arrow,
        "unpack": lambda raw: (
            _arrow_result_values(raw[0]).to_pylist(),
            _arrow_result_values(raw[1]).to_pylist(),
            _arrow_result_values(raw[2]).to_pylist(),
            raw[3],
            raw[4],
        ),
    },
    numpy_ops={
        "kernel": location_frequency_indexed_numpy,
        "unpack": lambda raw: (raw[0].tolist(), raw[1].tolist(), raw[2].tolist(), raw[3], raw[4]),
    },
)


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
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    ops = _DISPATCHER.get_ops(df)
    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"
    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)

    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    raw = ops["kernel"](lats_data, lngs_data, indices, ends)
    out_lats, out_lngs, out_counts_list, out_starts, out_ends = ops["unpack"](raw)

    freqs_all: list = []
    per_user_freqs: list = []
    uid_vals_all: list = []
    for i, (s, e) in enumerate(zip(out_starts.tolist(), out_ends.tolist())):
        seg = out_counts_list[s:e]
        total = sum(seg)
        if normalize:
            user_freqs = [c / total for c in seg]
        else:
            user_freqs = [float(c) for c in seg]
        freqs_all.extend(user_freqs)
        per_user_freqs.append(user_freqs)
        if uid_values is not None:
            uid_vals_all.extend([uid_values[i]] * len(seg))

    if uid_col is None:
        if as_ranks:
            return per_user_freqs[0]
        return _to_native({lat_col: out_lats, lng_col: out_lngs, "location_frequency": freqs_all}, df)

    if as_ranks:
        max_locs = max(len(f) for f in per_user_freqs) if per_user_freqs else 0
        ranks_lists: list = [[] for _ in range(max_locs)]
        for user_freqs in per_user_freqs:
            for rank_idx, freq in enumerate(user_freqs):
                ranks_lists[rank_idx].append(freq)
        return [sum(r) / len(r) for r in ranks_lists if r]

    return _to_native(
        {
            uid_col: uid_vals_all,
            lat_col: out_lats,
            lng_col: out_lngs,
            "location_frequency": freqs_all,
        },
        df,
    )
