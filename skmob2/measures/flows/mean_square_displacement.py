from __future__ import annotations

from datetime import timedelta
from typing import Any

from .._common import _build_user_ranges, _prepare_trajectory
from ..._core import square_displacement_km2


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

    For each user the square displacement is computed as the squared Haversine
    distance (in km) between their first recorded position ``r0`` and the last
    position whose timestamp satisfies ``datetime <= r0_time + delta_t``, where
    ``delta_t = timedelta(days=days, hours=hours, minutes=minutes)``.

    The function then returns the arithmetic mean of these per-user values —
    matching the skmob ``collective.mean_square_displacement`` formula exactly.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.  The
        trajectory must be sorted by datetime within each user (``_prepare_trajectory``
        performs this sort automatically).
    days:
        Days component of the time offset from each user's start time.
        Defaults to 0.
    hours:
        Hours component of the time offset.  Defaults to 1.
    minutes:
        Minutes component of the time offset.  Defaults to 0.
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
    float
        Mean square displacement in km².  Returns 0.0 when the trajectory is
        empty or every user's displacement window is trivially at the start.



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

    @usedBy
        skmob2.measures.flows.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    delta_t = timedelta(days=days, hours=hours, minutes=minutes)

    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    if len(df) == 0:
        return 0.0

    _uid_values, ranges = _build_user_ranges(df, uid_col)
    datetimes = df.get_column(datetime_col).to_list()
    lats = df.get_column(lat_col).to_list()
    lngs = df.get_column(lng_col).to_list()
    sq_displacements: list[float] = []
    for start, end in ranges:
        if start == end:
            continue
        t_limit = datetimes[start] + delta_t
        rt_idx = start
        for idx in range(start, end):
            if datetimes[idx] <= t_limit:
                rt_idx = idx
            else:
                break

        d2 = square_displacement_km2(
            float(lats[start]),
            float(lngs[start]),
            float(lats[rt_idx]),
            float(lngs[rt_idx]),
        )
        sq_displacements.append(d2)

    if not sq_displacements:
        return 0.0

    return sum(sq_displacements) / len(sq_displacements)
