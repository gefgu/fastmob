from __future__ import annotations

from typing import Any

import narwhals as nw
from .._common import _pick_existing_column, LAT_CANDIDATES, LNG_CANDIDATES
from ..individual.home_location import home_location


def homes_per_location(
    traj: Any,
    *,
    start_night: int = 22,
    end_night: int = 7,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the number of users whose home is at each distinct location.

    Calls :func:`~skmob2.measures.individual.home_location.home_location` to
    assign a home ``(lat, lng)`` to every user, then groups and counts.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
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

    Returns
    -------
    DataFrame
        One row per distinct home location with columns
        ``[lat_col, lng_col, "n_homes"]``, sorted by descending count.
        The returned backend matches the input backend.



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
    >>> from skmob2 import homes_per_location
    >>> result = homes_per_location(df)
    >>> print(result.round({"lat": 3, "lng": 3}).head().to_string(index=False))
       lat      lng  n_homes
    39.891 -105.069        1
    37.630 -122.411        1
    39.739 -104.985        1

    References
    ----------
    - [PRS2016] Pappalardo, L., Rinzivillo, S. & Simini, F. (2016) Human Mobility Modelling: exploration and preferential return meet the gravity model. Procedia Computer Science 83, 934-939, http://dx.doi.org/10.1016/j.procs.2016.04.188
    """
    home_df = home_location(
        traj,
        start_night=start_night,
        end_night=end_night,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    nw_home = nw.from_native(home_df, eager_only=True)

    cols = nw_home.columns
    detected_lat = _pick_existing_column(cols, LAT_CANDIDATES)
    detected_lng = _pick_existing_column(cols, LNG_CANDIDATES)

    if detected_lat is None or detected_lng is None:
        raise ValueError(
            f"Could not detect lat/lng columns in home_location result. "
            f"Looked for: lat={LAT_CANDIDATES}, lng={LNG_CANDIDATES}. "
            f"Available columns: {cols}."
        )
    result = (
        nw_home.select([detected_lat, detected_lng])
        .group_by([detected_lat, detected_lng])
        .agg(nw.len().alias("n_homes"))
        .sort("n_homes", descending=True)
    ).to_native()

    return result
