from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import uncorrelated_entropy_indexed
from fastmob.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _detect_trajectory_columns,
    _to_native,
)

_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def uncorrelated_entropy(
    traj: Any,
    *,
    normalize: bool = False,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the uncorrelated entropy of mobility for each user.

    Uncorrelated entropy is the Shannon entropy over the distribution of
    visit probabilities across distinct locations:

    \\[
    S_\\mathrm{unc} = -\\sum_i p_i \\log_2(p_i)
    \\]

    where \\(p_i\\) is the fraction of visits to location \\(i\\) out of the
    user's total visits.  A location is a unique exact ``(lat, lng)`` pair.

    When ``normalize=True`` the result is divided by \\(\\log_2(n)\\) (the
    random entropy) so the output lies in ``[0, 1]``.  If the user visits
    only one distinct location the entropy is 0; dividing by 0 is avoided
    by returning 0 directly.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    normalize : bool, optional
        When True, divide the Shannon entropy by \\(\\log_2(n_\\mathrm{distinct})\\)
        to normalise into ``[0, 1]``.  Default False.
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
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[uid_col, "uncorrelated_entropy"]``.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from fastmob import uncorrelated_entropy
    >>> result = uncorrelated_entropy(df)
    >>> print(result.round({"uncorrelated_entropy": 3}).head().to_string(index=False))
     uid  uncorrelated_entropy
       0                 7.442
       1                 3.650
       2                 6.908

    References
    ----------
    - [EP2009] Eagle, N. & Pentland, A. S. (2009) Eigenbehaviors: identifying structure in routine. Behavioral Ecology and Sociobiology 63(7), 1057-1066, https://link.springer.com/article/10.1007/s00265-009-0830-6
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018
    - [PVGSPG2016] Pappalardo, L., Vanhoof, M., Gabrielli, L., Smoreda, Z., Pedreschi, D. & Giannotti, F. (2016) An analytical framework to nowcast well-being using mobile phone data. International Journal of Data Science and Analytics 2(75), 75-92, https://link.springer.com/article/10.1007/s41060-016-0013-2

    See Also
    --------
    random_entropy : Maximum possible entropy assuming uniform visitation.
    real_entropy : Entropy that captures temporal order and frequency of visits.
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

    ops = _EXTRACTOR.get_ops(df)
    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col)

    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    entropies = _arrow_result_values(
        uncorrelated_entropy_indexed(lats_data, lngs_data, indices, ends, normalize)
    )

    if uid_col is None:
        return _to_native({"uncorrelated_entropy": entropies}, df)
    return _to_native({uid_col: uid_values, "uncorrelated_entropy": entropies}, df)
