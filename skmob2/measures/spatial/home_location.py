from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _is_pandas_backed, _prepare_trajectory


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

    @usedBy
        skmob2.measures.spatial.max_distance_from_home,
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    backend = df.implementation

    if _is_pandas_backed(df):
        pd_df = df.to_native()
        start_time = f"{start_night:02d}:00" if isinstance(start_night, int) else start_night
        end_time = f"{end_night:02d}:00" if isinstance(end_night, int) else end_night

        def _home_pandas(frame):
            night_visits = frame.set_index(pd.DatetimeIndex(frame[datetime_col])).between_time(start_time, end_time)
            source = night_visits if len(night_visits) != 0 else frame
            lat, lng = (
                source.groupby([lat_col, lng_col])
                .count()
                .sort_values(by=datetime_col, ascending=False)
                .iloc[0]
                .name
            )
            return lat, lng

        import pandas as pd

        if uid_col is None:
            lat, lng = _home_pandas(pd_df)
            return pd.DataFrame({lat_col: [lat], lng_col: [lng]})

        rows = []
        for uid, group in pd_df.groupby(uid_col):
            lat, lng = _home_pandas(group)
            rows.append({uid_col: uid, lat_col: lat, lng_col: lng})
        return pd.DataFrame(rows, columns=[uid_col, lat_col, lng_col])

    # Extract hour from datetime column via Narwhals dt accessor.
    df = df.with_columns(nw.col(datetime_col).dt.hour().alias("__hour__"))

    # Build nighttime mask: hours >= start_night OR hours < end_night.
    night_mask = (nw.col("__hour__") >= start_night) | (nw.col("__hour__") < end_night)
    night_df = df.filter(night_mask)

    def _most_visited(frame: nw.DataFrame, group_keys: list[str]) -> nw.DataFrame:
        """Return the most-visited (lat, lng) per group, matching skmob ties."""
        visit_counts = frame.group_by(group_keys + [lat_col, lng_col]).agg(
            nw.len().alias("__count__"),
        )
        if uid_col is not None:
            # skmob uses pandas groupby(...).count().sort_values(count, descending)
            # and then iloc[0], so equal-count ties follow groupby's sorted
            # coordinate order.
            best_per_user = (
                visit_counts.sort([uid_col, "__count__", lat_col, lng_col], descending=[False, True, False, False])
                .group_by([uid_col])
                .agg(
                    nw.col(lat_col).first().alias(lat_col),
                    nw.col(lng_col).first().alias(lng_col),
                )
            )
            return best_per_user.select([uid_col, lat_col, lng_col])
        else:
            best = visit_counts.sort(["__count__", lat_col, lng_col], descending=[True, False, False]).rows(named=True)[0]
            return nw.from_dict(
                {lat_col: [best[lat_col]], lng_col: [best[lng_col]]},
                backend=backend,
            )

    if uid_col is not None:
        group_keys = [uid_col]
    else:
        group_keys = []

    # Try nighttime records first; fall back to all-hours for users with no night data.
    if len(night_df) == 0:
        # No nighttime records at all — use full dataset.
        return _most_visited(df, group_keys).to_native()

    night_result = _most_visited(night_df, group_keys)

    if uid_col is None:
        # Single-user path; no fallback needed.
        return night_result.to_native()

    # Find users that had no nighttime records and fill in from all-hours fallback.
    all_uids = set(df.get_column(uid_col).to_list())
    night_uids = set(night_result.get_column(uid_col).to_list())
    missing_uids = all_uids - night_uids

    if not missing_uids:
        return night_result.to_native()

    fallback_df = df.filter(nw.col(uid_col).is_in(list(missing_uids)))
    fallback_result = _most_visited(fallback_df, group_keys)

    # Combine night result and fallback result.
    combined = nw.concat([night_result, fallback_result])
    return combined.to_native()
