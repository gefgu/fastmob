from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def visits_per_location(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total number of visits for each distinct location.

    Counts all trajectory rows (visits) per unique ``(lat, lng)`` pair across
    all users.  This is the population-level analogue of per-user
    ``location_frequency``.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None (not used for
        aggregation, but retained for consistent preprocessing).

    Returns
    -------
    DataFrame
        One row per distinct ``(lat, lng)`` location with columns
        ``[lat_col, lng_col, "n_visits"]``, sorted by descending visit count.
        The returned backend matches the input backend.


    References
    ----------
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

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

    result = (
        df.select([lat_col, lng_col])
        .group_by([lat_col, lng_col])
        .agg(nw.len().alias("n_visits"))
        .sort("n_visits", descending=True)
    )

    return result.to_native()
