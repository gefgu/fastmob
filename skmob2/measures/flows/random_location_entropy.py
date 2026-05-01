from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def random_location_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the random entropy for each distinct location across all users.

    Random location entropy is \\(\\log_2(n)\\), where \\(n\\) is the number of
    distinct users who visited the location.  A location is a unique exact
    ``(lat, lng)`` pair — matching the skmob convention.

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
        ``[lat_col, lng_col, "random_entropy"]``.
        The returned backend matches the input backend.

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
        # Single user: each location is visited by exactly 1 user -> entropy = 0.
        locs = df.select([lat_col, lng_col]).unique().sort([lat_col, lng_col])
        result = locs.with_columns(nw.lit(0.0).alias("random_entropy"))
        return result.to_native()

    # Count distinct users per (lat, lng) location.
    grouped = (
        df.select([uid_col, lat_col, lng_col])
        .unique()  # one row per (uid, lat, lng) combination
        .group_by([lat_col, lng_col])
        .agg(nw.col(uid_col).n_unique().alias("__n_users__"))
        .sort([lat_col, lng_col])
    )

    n_users_list = grouped.get_column("__n_users__").to_list()
    entropies = [math.log2(n) if n > 1 else 0.0 for n in n_users_list]

    lats = grouped.get_column(lat_col).to_list()
    lngs = grouped.get_column(lng_col).to_list()

    return nw.from_dict(
        {lat_col: lats, lng_col: lngs, "random_entropy": entropies},
        backend=backend,
    ).to_native()
