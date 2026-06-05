from __future__ import annotations
import narwhals as nw

from typing import Any

from skmob2._core import (
    home_location_arrow,
    home_location_indexed_arrow,
    home_location_indexed_numpy,
    home_location_numpy,
)

from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ranges,
    _dispatch_pair_kernel,
    _extract_hours,
    _detect_trajectory_columns,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def home_location(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
) -> Any:
    """Return the most-visited nighttime location for each user.

    The home location is the (lat, lng) pair visited most often during the
    nighttime window ``[start_night, 24) ∪ [0, end_night)``.  When a user
    has no nighttime records, the most-visited location across all hours is
    used as the fallback.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    start_night:
        Hour (0–23) at which the nighttime window begins.  Default: 22.
    end_night:
        Hour (0–23) at which the nighttime window ends (exclusive).  Default: 7.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.
    sorted:
        When True, trust that rows are already grouped by user and use the
        contiguous fast path.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, lat_col, lng_col]`` (using
        the detected column names).  The returned backend matches the input
        backend.



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
    >>> from skmob2 import home_location
    >>> result = home_location(df)
    >>> print(result.round({"lat": 3, "lng": 3}).head().to_string(index=False))
     uid    lat      lng
       0 39.891 -105.069
       1 37.630 -122.411
       2 39.739 -104.985

    References
    ----------
    - [CBTDHVSB2012] Csaji, B. C., Browet, A., Traag, V. A., Delvenne, J.-C., Huens, E., Van Dooren, P., Smoreda, Z. & Blondel, V. D. (2012) Exploring the Mobility of Mobile Phone Users. Physica A: Statistical Mechanics and its Applications 392(6), 1459-1473, https://www.sciencedirect.com/science/article/pii/S0378437112010059
    - [PSO2012] Phithakkitnukoon, S., Smoreda, Z. & Olivier, P. (2012) Socio-geography of human mobility: A study using longitudinal mobile phone data. PLOS ONE 7(6): e39253. https://doi.org/10.1371/journal.pone.0039253
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

    df, hours = _extract_hours(df, datetime_col)
    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"
    if sorted:
        uid_values, ranges = _build_presorted_user_ranges(df, uid_col)
        home_lats, home_lngs = _dispatch_pair_kernel(
            home_location_numpy,
            home_location_arrow,
            [df.get_column(lat_col), df.get_column(lng_col), hours],
            ranges,
            float(start_night),
            float(end_night),
            use_arrow=use_arrow,
        )
        if uid_col is None:
            return _to_native({lat_col: home_lats, lng_col: home_lngs}, df)
        return _to_native({uid_col: uid_values, lat_col: home_lats, lng_col: home_lngs}, df)

    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)

    home_lats, home_lngs = _dispatch_pair_kernel(
        home_location_indexed_numpy,
        home_location_indexed_arrow,
        [df.get_column(lat_col), df.get_column(lng_col), hours],
        indices,
        ends,
        float(start_night),
        float(end_night),
        use_arrow=use_arrow,
    )

    if uid_col is None:
        return _to_native({lat_col: home_lats, lng_col: home_lngs}, df)
    return _to_native({uid_col: uid_values, lat_col: home_lats, lng_col: home_lngs}, df)
