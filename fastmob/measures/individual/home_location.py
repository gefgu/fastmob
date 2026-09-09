from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob._core import home_location_indexed, home_location_presorted
from fastmob.utils._common import (
    _as_arrow,
    _build_user_ranges_auto,
    _detect_trajectory_columns,
    _extract_hours,
    _to_native,
)


def _format_pair(values: tuple[Any, Any]) -> tuple[Any, Any]:
    first, second = values
    return _as_arrow(first), _as_arrow(second)


def home_location(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the most-visited nighttime location for each user.

    The home location :math:`h(u)` of an individual :math:`u` is the location
    visited most often during the nighttime window
    ``[start_night, 24) ∪ [0, end_night)`` [CBTDHVSB2012]_ [PSO2012]_:

    .. math::

        h(u) = \\arg\\max_{i}
            \\bigl|\\{r_i \\mid t(r_i) \\in [t_{\\text{start}}, t_{\\text{end}}]\\}\\bigr|

    where :math:`r_i` is a location visited by :math:`u`, :math:`t(r_i)` is
    the time of the visit, and :math:`t_{\\text{start}}` / :math:`t_{\\text{end}}`
    bound the nighttime window.  When a user has no nighttime records, the
    most-visited location across all hours is used as a fallback.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    start_night : int, optional
        Hour (0–23) at which the nighttime window begins.  Default: 22.
    end_night : int, optional
        Hour (0–23) at which the nighttime window ends (exclusive).  Default: 7.
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
        One row per user with columns ``[uid_col, lat_col, lng_col]`` (using
        the detected column names).  The returned backend matches the input
        backend.

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
    >>> from fastmob import home_location
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

    See Also
    --------
    max_distance_from_home : Maximum distance from the inferred home location.
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

    df, hours = _extract_hours(df, datetime_col)
    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    hours_data = hours.to_arrow()
    uid_values, indices, ends = _build_user_ranges_auto(df, uid_col, coordinates=(lats_data, lngs_data))
    if indices is None:
        home_lats, home_lngs = _format_pair(
            home_location_presorted(
                lats_data,
                lngs_data,
                hours_data,
                ends,
                float(start_night),
                float(end_night),
            )
        )
        if uid_col is None:
            return _to_native({lat_col: home_lats, lng_col: home_lngs}, df)
        return _to_native({uid_col: uid_values, lat_col: home_lats, lng_col: home_lngs}, df)

    home_lats, home_lngs = _format_pair(
        home_location_indexed(
            lats_data,
            lngs_data,
            hours_data,
            indices,
            ends,
            float(start_night),
            float(end_night),
        )
    )

    if uid_col is None:
        return _to_native({lat_col: home_lats, lng_col: home_lngs}, df)
    return _to_native({uid_col: uid_values, lat_col: home_lats, lng_col: home_lngs}, df)
