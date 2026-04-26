from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


def number_of_locations(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the number of distinct locations visited by each user.

    A distinct location is a unique exact ``(lat, lng)`` pair — matching the
    skmob convention of float equality without spatial clustering.

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
        One row per user with columns ``[uid_col, "number_of_locations"]``.
        The returned backend matches the input backend.

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

    # Build a single string key per row encoding the (lat, lng) pair so that
    # Narwhals n_unique() can count distinct locations without nw.struct support.
    loc_key_col = "__skmob2_loc_key__"
    df = df.with_columns(
        (nw.col(lat_col).cast(nw.String) + nw.lit("_") + nw.col(lng_col).cast(nw.String)).alias(loc_key_col)
    )

    if uid_col is None:
        n_locs = df.get_column(loc_key_col).n_unique()
        return nw.from_dict(
            {"number_of_locations": [n_locs]},
            backend=df.implementation,
        ).to_native()

    result = df.group_by(uid_col).agg(nw.col(loc_key_col).n_unique().alias("number_of_locations")).sort(uid_col)

    return result.to_native()
