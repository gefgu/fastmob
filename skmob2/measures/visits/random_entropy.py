from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def random_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the random entropy of mobility for each user.

    Random entropy is defined as \\(\\log_2(n)\\), where \\(n\\) is the number of
    distinct locations visited by the user.  A location is a unique exact
    ``(lat, lng)`` pair — matching the skmob convention (float equality,
    no spatial clustering).

    This is the maximum possible entropy for a user who visits \\(n\\)
    distinct places, assuming all locations are equally likely.

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
        One row per user with columns ``[uid_col, "random_entropy"]``.
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
    >>> from skmob2 import random_entropy
    >>> result = random_entropy(df)
    >>> print(result.round({"random_entropy": 3}).head().to_string(index=False))
     uid  random_entropy
       0           9.082
       1           6.600
       2           8.738

    References
    ----------
    - [EP2009] Eagle, N. & Pentland, A. S. (2009) Eigenbehaviors: identifying structure in routine. Behavioral Ecology and Sociobiology 63(7), 1057-1066, https://link.springer.com/article/10.1007/s00265-009-0830-6
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018

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

    # Encode each (lat, lng) pair as a single string key for n_unique counting.
    loc_key_col = "__skmob2_loc_key__"
    df = df.with_columns(
        (nw.col(lat_col).cast(nw.String) + nw.lit("_") + nw.col(lng_col).cast(nw.String)).alias(loc_key_col)
    )

    if uid_col is None:
        n_locs = df.get_column(loc_key_col).n_unique()
        entropy = math.log2(n_locs) if n_locs > 1 else 0.0
        return nw.from_dict(
            {"random_entropy": [entropy]},
            backend=df.implementation,
        ).to_native()

    # Group by user, count distinct locations, apply log2 in Python.
    grouped = df.group_by(uid_col).agg(nw.col(loc_key_col).n_unique().alias("__n_locs__")).sort(uid_col)

    uid_vals = grouped.get_column(uid_col).to_list()
    n_locs_vals = grouped.get_column("__n_locs__").to_list()
    entropies = [math.log2(n) if n > 1 else 0.0 for n in n_locs_vals]

    return nw.from_dict(
        {uid_col: uid_vals, "random_entropy": entropies},
        backend=df.implementation,
    ).to_native()
