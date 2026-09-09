from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob._core import location_frequency_presorted, location_frequency_values_indexed
from fastmob.utils._common import (
    _as_arrow,
    _build_user_ranges_auto,
    _detect_trajectory_columns,
    _take_uid_values,
    _to_native,
    _values_to_list,
)


def _unpack_location_frequency(raw: tuple[Any, ...]) -> tuple[Any, ...]:
    return (
        _as_arrow(raw[0]),
        _as_arrow(raw[1]),
        _as_arrow(raw[2]),
        raw[3],
        raw[4],
        raw[5],
        _as_arrow(raw[6]),
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

    The visitation frequency :math:`f(r_i)` of location :math:`r_i` for
    individual :math:`u` is the probability of visiting that location
    [SKWB2010]_ [PF2018]_:

    .. math::

        f(r_i) = \\frac{n(r_i)}{n_u}

    where :math:`n(r_i)` is the number of visits to location :math:`r_i` by
    :math:`u`, and :math:`n_u` is the total number of data points in
    :math:`u`'s trajectory.  When ``normalize=False``, raw visit counts
    :math:`n(r_i)` are returned instead.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    normalize : bool, optional
        When ``True`` (default), the ``"location_frequency"`` column contains
        the probability of visiting that location (count / total_visits), so
        each user's frequencies sum to 1.0.  When ``False``, raw visit counts
        are returned.
    as_ranks : bool, optional
        When ``True``, return a Python list where element *i* is the mean
        visit frequency of the *i*-th most-visited location across all users
        (rank-1 = most visited).  The list length equals the maximum number
        of distinct locations visited by any single user.  When ``False``
        (default), return a DataFrame.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.
    Returns
    -------
    pandas.DataFrame or polars.DataFrame or list
        When ``as_ranks=False``: one row per ``(user, location)`` pair with
        columns ``[uid_col, lat_col, lng_col, "location_frequency"]``.
        When ``as_ranks=True``: a flat Python ``list[float]`` of mean
        per-rank frequencies.
        The returned DataFrame backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.data.BRIGHTKITE_SAMPLE
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
    >>> from fastmob import location_frequency
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
    - [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

    See Also
    --------
    frequency_rank : Rank locations by visit frequency (1 = most visited).
    visits_per_location : Total visits per location across all users (collective measure).
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
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

    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()

    uid_values, indices, ends = _build_user_ranges_auto(df, uid_col, coordinates=(lats_data, lngs_data))
    if indices is None:
        raw = location_frequency_presorted(lats_data, lngs_data, ends, normalize)
    else:
        raw = location_frequency_values_indexed(lats_data, lngs_data, indices, ends, normalize)
    out_lats, out_lngs, freqs, user_indices, _out_starts, _out_ends, rank_means = _unpack_location_frequency(raw)

    if as_ranks:
        return _values_to_list(rank_means)

    if uid_col is None:
        return _to_native({lat_col: out_lats, lng_col: out_lngs, "location_frequency": freqs}, df)

    return _to_native(
        {
            uid_col: _take_uid_values(uid_values, user_indices),
            lat_col: out_lats,
            lng_col: out_lngs,
            "location_frequency": freqs,
        },
        df,
    )
