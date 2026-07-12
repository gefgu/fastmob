"""Real entropy of individual mobility trajectories (Kontoyiannis estimator)."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import (
    real_entropy_indexed_arrow,
    real_entropy_indexed_numpy,
)
from fastmob.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_ms,
    _with_datetime_column,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "real_entropy_indexed": real_entropy_indexed_arrow,
        "format_result": lambda values: _arrow_result_values(values).to_pylist(),
    },
    numpy_ops={
        "real_entropy_indexed": real_entropy_indexed_numpy,
        "format_result": lambda values: values.tolist(),
    },
)


def real_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the real (true) entropy of mobility for each user.

    Real entropy is estimated using the Kontoyiannis (1998) Lempel-Ziv
    entropy rate estimator applied to the sequence of visited locations.
    Each location is encoded as the exact ``(lat, lng)`` float pair using
    bitwise equality — matching the skmob convention (no spatial clustering).

    The estimator captures both the frequency and the order of visits,
    unlike random entropy (which ignores order) and uncorrelated entropy
    (which ignores temporal correlations).

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per user with columns ``[uid_col, "real_entropy"]``.
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
    >>> from fastmob import real_entropy
    >>> result = real_entropy(df)
    >>> print(result.round({"real_entropy": 3}).head().to_string(index=False))
     uid  real_entropy
       0         4.905
       1         2.200
       2         4.683

    References
    ----------
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018

    See Also
    --------
    random_entropy : Maximum possible entropy assuming uniform visitation.
    uncorrelated_entropy : Entropy weighted by visit frequency (ignores temporal order).
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = _with_datetime_column(df, datetime_col)
    df = df.drop_nulls(subset=[datetime_col]).with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    ops = _DISPATCHER.get_ops(df)
    timestamps = _extract_timestamps_ms(df, datetime_col)
    timestamps_data = ops["extract_data"](timestamps)
    uid_values, indices, ends = _build_time_ordered_user_ranges(
        df, uid_col, datetime_col, timestamps_data
    )

    raw = ops["real_entropy_indexed"](
        ops["extract_data"](df.get_column(lat_col)),
        ops["extract_data"](df.get_column(lng_col)),
        indices,
        ends,
    )
    entropies = ops["format_result"](raw)

    result_dict: dict[str, Any] = {"real_entropy": entropies}
    if uid_col is not None:
        result_dict[uid_col] = uid_values

    return nw.from_dict(result_dict, backend=df.implementation).to_native()
