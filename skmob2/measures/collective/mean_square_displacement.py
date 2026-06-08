from __future__ import annotations
import narwhals as nw

from datetime import timedelta
from typing import Any

from skmob2._core import (
    mean_square_displacement_indexed_arrow,
    mean_square_displacement_indexed_numpy,
)
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _build_time_ordered_user_ranges,
    _extract_timestamps_s,
    _detect_trajectory_columns,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={"kernel": mean_square_displacement_indexed_arrow},
    numpy_ops={"kernel": mean_square_displacement_indexed_numpy},
)


def mean_square_displacement(
    traj: Any,
    *,
    days: int = 0,
    hours: int = 1,
    minutes: int = 0,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> float:
    """Return the mean square displacement (km²) across all users.

    The mean squared displacement (MSD) measures the average deviation of
    position from a reference point over time [FS2002]_ [BHG2006]_
    [SKWB2010]_:

    .. math::

        \\text{MSD}(t) = \\frac{1}{N} \\sum_{i=1}^{N}
            \\bigl|r^{(i)}(t) - r^{(i)}(0)\\bigr|^2

    where :math:`N` is the number of individuals, :math:`r^{(i)}(0)` is the
    reference position (first recorded point) of individual :math:`i`, and
    :math:`r^{(i)}(t)` is their position at time offset :math:`t`.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    days : int, optional
        Days component of the time offset from each user's start time.
        Defaults to 0.
    hours : int, optional
        Hours component of the time offset.  Defaults to 1.
    minutes : int, optional
        Minutes component of the time offset.  Defaults to 0.
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
    float
        Mean square displacement in km².  Returns 0.0 when the trajectory is
        empty or every user's displacement window is trivially at the start.

    Warning
    -------
    The trajectory must be sorted in ascending order by datetime within each
    user for the displacement window to be computed correctly.

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
    >>> from skmob2 import mean_square_displacement
    >>> result = mean_square_displacement(df)
    >>> print(round(result, 3))
    128.583

    References
    ----------
    - [FS2002] Frenkel, D. & Smit, B. (2002) Understanding molecular simulation: From algorithms to applications. Academic Press, 196 (2nd Ed.), https://www.sciencedirect.com/book/9780122673511/understanding-molecular-simulation.
    - [BHG2006] Brockmann, D., Hufnagel, L. & Geisel, T. (2006) The scaling laws of human travel. Nature 439, 462-465, https://www.nature.com/articles/nature04292
    - [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    """
    delta_s = timedelta(days=days, hours=hours, minutes=minutes).total_seconds()

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

    if len(df) == 0:
        return 0.0

    ops = _DISPATCHER.get_ops(df)
    timestamps = _extract_timestamps_s(df, datetime_col)
    timestamps_data = ops["extract_data"](timestamps)
    _uid_values, indices, ends = _build_time_ordered_user_ranges(
        df, uid_col, datetime_col, timestamps_data
    )

    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    return ops["kernel"](lats_data, lngs_data, timestamps_data, indices, ends, delta_s)
