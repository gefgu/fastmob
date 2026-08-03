"""Activity-purpose inference: dwell-time activity flagging and home/work
location identification for Staypoints/Locations.

Ported from trackintel's `create_activity_flag` (`analysis/labelling.py`)
and a scoped version of `location_identifier(method="freq")`
(`analysis/location_identification.py`), plus transbigdata's
`mobile_identify_home`/`mobile_identify_work` night/day-duration heuristic.
Operates on the small, already-aggregated Staypoints/Locations tables (one
row per stop or per user-location, never per raw GPS fix) -- plain
Narwhals, no new Rust kernel needed, same precedent as `_transport_mode.py`.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw


def _as_native_and_wrapper(staypoints: Any) -> tuple[Any, Any]:
    """Return ``(native_df, staypoints_or_None)`` -- mirrors
    `_transport_mode._as_native_and_wrapper`.
    """
    if hasattr(staypoints, "df"):
        return staypoints.df, staypoints
    return staypoints, None


def create_activity_flag(
    staypoints: Any,
    method: str = "time_threshold",
    time_threshold_min: float = 15.0,
) -> Any:
    """Flag each staypoint as a genuine "activity" (vs. a transient stop) by dwell time.

    Parameters
    ----------
    staypoints : Staypoints or DataFrame-like
        Must have ``started_at``/``finished_at`` columns (or the
        `Staypoints` wrapper's own resolved column names).
    method : str, optional
        Only ``"time_threshold"`` is implemented.
    time_threshold_min : float, optional
        Minimum dwell time, in minutes, to count as an activity.
        Default ``15.0``.

    Returns
    -------
    Staypoints or DataFrame-like
        Input with an added boolean ``activity`` column.

    References
    ----------
    - Martin, F. et al. (2023) trackintel: ``create_activity_flag``.
    """
    if method != "time_threshold":
        raise ValueError(f"unknown activity-flag method: {method!r}; choose from ['time_threshold']")

    native_df, wrapper = _as_native_and_wrapper(staypoints)
    started_at_col = getattr(wrapper, "started_at_col", "started_at")
    finished_at_col = getattr(wrapper, "finished_at_col", "finished_at")

    nw_df = nw.from_native(native_df, eager_only=True)
    duration_min = (nw.col(finished_at_col).dt.timestamp("ms") - nw.col(started_at_col).dt.timestamp("ms")) / 60000.0
    result_native = nw_df.with_columns((duration_min >= time_threshold_min).alias("activity")).to_native()

    if wrapper is not None:
        from ..core.staypoints_dataframe import Staypoints

        return Staypoints(
            result_native,
            uid_col=wrapper.uid_col,
            lat_col=wrapper.lat_col,
            lng_col=wrapper.lng_col,
            started_at_col=started_at_col,
            finished_at_col=finished_at_col,
            parameters=wrapper.parameters,
        )
    return result_native


create_activity_flag.__module__ = "fastmob.preprocessing"


def identify_locations(
    locations: Any,
    staypoints: Any,
    method: str = "freq",
    night_start_hour: int = 22,
    night_end_hour: int = 6,
    work_start_hour: int = 8,
    work_end_hour: int = 18,
) -> Any:
    """Label each user's recurring `Locations` as ``"home"``, ``"work"``, or ``"other"``.

    Parameters
    ----------
    locations : Locations
        Locations previously generated from ``staypoints`` (via
        `Staypoints.generate_user_locations`), so ``location_id`` values line up.
    staypoints : Staypoints
        Must carry a ``location_id`` column (the enriched `Staypoints`
        returned alongside ``locations``).
    method : str, optional
        Only ``"freq"`` is implemented: per user, the location with the
        most total night-time dwell (``night_start_hour``..``night_end_hour``,
        wrapping midnight) is labeled ``"home"``; the location (other than
        home) with the most total weekday daytime dwell
        (``work_start_hour``..``work_end_hour``, Mon-Fri) is labeled
        ``"work"``. Every other location is ``"other"``.
    night_start_hour, night_end_hour, work_start_hour, work_end_hour : int, optional
        Hour-of-day boundaries (0-23) used by the ``"freq"`` heuristic.

    Returns
    -------
    Locations
        A copy of ``locations`` with an added ``purpose`` column.

    References
    ----------
    - Martin, F. et al. (2023) trackintel: ``location_identifier(method="FREQ")``.
    - transbigdata: ``mobile_identify_home``/``mobile_identify_work``.
    """
    if method != "freq":
        raise ValueError(f"unknown location-identification method: {method!r}; choose from ['freq']")

    from ..core.locations_dataframe import Locations

    if locations.scope != "user":
        raise ValueError("identify_locations only supports user-scoped Locations; purposes are user-specific")

    uid_col = staypoints.uid_col
    if uid_col is None:
        raise ValueError("identify_locations requires staypoints to have a uid column")

    sp_nw = nw.from_native(staypoints.df, eager_only=True)
    if "location_id" not in sp_nw.columns:
        raise ValueError("identify_locations requires staypoints to have a 'location_id' column")

    started_at_col = staypoints.started_at_col
    finished_at_col = staypoints.finished_at_col

    duration_min = (nw.col(finished_at_col).dt.timestamp("ms") - nw.col(started_at_col).dt.timestamp("ms")) / 60000.0
    hour = nw.col(started_at_col).dt.hour()
    weekday = nw.col(started_at_col).dt.weekday()

    is_night = (hour >= night_start_hour) | (hour < night_end_hour)
    is_workday_daytime = (weekday <= 5) & (hour >= work_start_hour) & (hour < work_end_hour)

    flagged = sp_nw.filter(~nw.col("location_id").is_null()).with_columns(
        (duration_min * is_night.cast(nw.Float64)).alias("__night_min__"),
        (duration_min * is_workday_daytime.cast(nw.Float64)).alias("__work_min__"),
    )

    per_loc = flagged.group_by([uid_col, "location_id"]).agg(
        nw.col("__night_min__").sum().alias("night_min"),
        nw.col("__work_min__").sum().alias("work_min"),
    )

    home = (
        per_loc.with_columns(nw.col("night_min").max().over(uid_col).alias("__max_night__"))
        .filter((nw.col("night_min") == nw.col("__max_night__")) & (nw.col("night_min") > 0))
        .group_by(uid_col)
        .agg(nw.col("location_id").min().alias("home_location_id"))
    )

    work_candidates = per_loc.join(home, on=uid_col, how="left").filter(
        nw.col("location_id") != nw.col("home_location_id").fill_null(-1)
    )
    work = (
        work_candidates.with_columns(nw.col("work_min").max().over(uid_col).alias("__max_work__"))
        .filter((nw.col("work_min") == nw.col("__max_work__")) & (nw.col("work_min") > 0))
        .group_by(uid_col)
        .agg(nw.col("location_id").min().alias("work_location_id"))
    )

    locations_nw = nw.from_native(locations.df, eager_only=True)
    labeled = (
        locations_nw.join(home, on=uid_col, how="left")
        .join(work, on=uid_col, how="left")
        .with_columns(
            nw.when(nw.col("location_id") == nw.col("home_location_id"))
            .then(nw.lit("home"))
            .when(nw.col("location_id") == nw.col("work_location_id"))
            .then(nw.lit("work"))
            .otherwise(nw.lit("other"))
            .alias("purpose")
        )
        .drop("home_location_id", "work_location_id")
    )

    return Locations(labeled.to_native(), uid_col=locations.uid_col, scope="user", scheme=locations.scheme)


identify_locations.__module__ = "fastmob.preprocessing"
