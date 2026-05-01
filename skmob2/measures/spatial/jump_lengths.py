from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    jump_lengths_arrow,
    jump_lengths_flat_arrow,
    jump_lengths_flat_numpy,
    jump_lengths_numpy,
)

from .._common import _build_user_ranges, _is_polars_backed, _prepare_trajectory


def _route_jump_lengths(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
    merge: bool,
) -> list[float] | list[list[float]]:
    if use_arrow:
        if merge:
            return jump_lengths_flat_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)
        return jump_lengths_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    if merge:
        return jump_lengths_flat_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)
    return jump_lengths_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def jump_lengths(
    traj: Any,
    merge: bool = False,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
):
    """Compute jump lengths (km) for each user in the trajectory.

    A *jump length* is the Haversine distance (in km) between consecutive
    GPS fixes for the same user, sorted by datetime.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    merge:
        When True, return a flat ``list[float]`` of all jump lengths across
        all users.  When False (default), return a per-user dataframe.
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
    DataFrame | list[float]
        When ``merge=False``: a dataframe with columns ``[uid_col, "jump_lengths"]``
        where each row holds a list of jump lengths for one user.  The returned
        backend matches the input backend.
        When ``merge=True``: a flat ``list[float]`` of all jump lengths.



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
    >>> from skmob2 import jump_lengths
    >>> result = jump_lengths(df)
    >>> preview = result.assign(n_jumps=result["jump_lengths"].str.len())
    >>> print(preview[["uid", "n_jumps"]].head().to_string(index=False))
     uid  n_jumps
       0     2098
       1     1209
       2     1690

    References
    ----------
    - [BHG2006] Brockmann, D., Hufnagel, L. & Geisel, T. (2006) The scaling laws of human travel. Nature 439, 462-465, https://www.nature.com/articles/nature04292
    - [GHB2008] Gonzalez, M. C., Hidalgo, C. A. & Barabasi, A. L. (2008) Understanding individual human mobility patterns. Nature, 453, 779-782, https://www.nature.com/articles/nature06958.
    - [PRQPG2013] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D. & Giannotti, F. (2013) Understanding the patterns of car travel. European Physics Journal Special Topics 215(1), 61-73, https://link.springer.com/article/10.1140%2Fepjst%2Fe2013-01715-5


    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        ranges = [(0, len(df))]
        jump_values = _route_jump_lengths(lats_full, lngs_full, ranges, use_arrow=use_arrow, merge=merge)
        if merge:
            return jump_values
        return nw.from_dict({"jump_lengths": jump_values}, backend=df.implementation).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    jump_values = _route_jump_lengths(lats_full, lngs_full, ranges, use_arrow=use_arrow, merge=merge)
    if merge:
        return jump_values

    result = nw.from_dict(
        {uid_col: uid_values, "jump_lengths": jump_values},
        backend=df.implementation,
    )

    return result.to_native()
