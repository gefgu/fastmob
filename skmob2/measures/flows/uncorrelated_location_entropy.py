from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory, _shannon_entropy


def uncorrelated_location_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the uncorrelated entropy for each distinct location across all users.

    For each location ``l``, computes the Shannon entropy over the distribution
    of visit probabilities:

    \\[
    S_\\mathrm{unc}(l) = -\\sum_u p(u, l) \\log_2(p(u, l))
    \\]

    where \\(p(u, l)\\) is the fraction of all visits to \\(l\\) that belong to
    user \\(u\\).  A location is a unique exact ``(lat, lng)`` pair.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent all rows are treated as
        coming from a single individual (entropy will be 0 everywhere).
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
        One row per distinct ``(lat, lng)`` location with columns
        ``[lat_col, lng_col, "uncorrelated_entropy"]``.
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
    >>> from skmob2 import uncorrelated_location_entropy
    >>> result = uncorrelated_location_entropy(df)
    >>> print(result.round({"uncorrelated_entropy": 3}).head().to_string(index=False))
          lat        lng  uncorrelated_entropy
     0.000000   0.000000                 0.911
    29.532220 -98.300914                 0.000
    29.942673 -90.064455                 0.000
    29.948116 -90.063436                 0.000
    29.948125 -90.063510                 0.000

    References
    ----------
    - [CML2011] Cho, E., Myers, S. A. & Leskovec, J. (2011) Friendship and mobility: user movement in location-based social networks. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, 1082-1090, https://dl.acm.org/citation.cfm?id=2020579

    @usedBy
        skmob2.measures.flows.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    backend = df.implementation

    if uid_col is None:
        # Single user: each location has probability 1 -> entropy = 0.
        locs = df.select([lat_col, lng_col]).unique().sort([lat_col, lng_col])
        result = locs.with_columns(nw.lit(0.0).alias("uncorrelated_entropy"))
        return result.to_native()

    # Count visits per (uid, lat, lng) triplet, then collect into Python dicts
    # keyed by location so we can compute Shannon entropy without per-location
    # dataframe filter passes (avoids O(L × N) work).
    visit_counts = (
        df.select([uid_col, lat_col, lng_col])
        .group_by([uid_col, lat_col, lng_col])
        .agg(nw.len().alias("__visits__"))
        .sort([lat_col, lng_col])
    )

    lat_list = visit_counts.get_column(lat_col).to_list()
    lng_list = visit_counts.get_column(lng_col).to_list()
    cnt_list = visit_counts.get_column("__visits__").to_list()

    # Accumulate per-location visit-count vectors: {(lat, lng): [c_u1, c_u2, ...]}
    loc_counts: dict[tuple, list[int]] = {}
    for lat_v, lng_v, c in zip(lat_list, lng_list, cnt_list):
        key = (lat_v, lng_v)
        loc_counts.setdefault(key, []).append(c)

    # Sort locations for stable output order.
    sorted_locs = sorted(loc_counts.keys())
    loc_lats = [k[0] for k in sorted_locs]
    loc_lngs = [k[1] for k in sorted_locs]
    entropies = [_shannon_entropy(loc_counts[k]) for k in sorted_locs]

    return nw.from_dict(
        {lat_col: loc_lats, lng_col: loc_lngs, "uncorrelated_entropy": entropies},
        backend=backend,
    ).to_native()
