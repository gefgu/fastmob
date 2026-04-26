from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _prepare_trajectory


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

    # Extract hour from datetime column via Narwhals dt accessor.
    df = df.with_columns(nw.col(datetime_col).dt.hour().alias("__hour__"))

    # Build nighttime mask: hours >= start_night OR hours < end_night.
    night_mask = (nw.col("__hour__") >= start_night) | (nw.col("__hour__") < end_night)
    night_df = df.filter(night_mask)

    # For each user, count visits per (lat, lng) location and record the first
    # row index of each location so ties can be broken deterministically.
    def _most_visited(frame: nw.DataFrame, group_keys: list[str]) -> nw.DataFrame:
        """Return the most-visited (lat, lng) per group, ties broken by first occurrence."""
        # Attach a sequential index so we can identify the first occurrence of each location.
        frame = frame.with_row_index("__loc_idx__")
        visit_counts = frame.group_by(group_keys + [lat_col, lng_col]).agg(
            nw.len().alias("__count__"),
            nw.col("__loc_idx__").min().alias("__first_idx__"),
        )
        if uid_col is not None:
            # For each user find the max visit count, then pick the location with
            # that count and the smallest first-occurrence index (stable tie-break).
            max_counts = visit_counts.group_by([uid_col]).agg(nw.col("__count__").max().alias("__max_count__"))
            best_per_user = (
                visit_counts.join(max_counts, on=[uid_col])
                .filter(nw.col("__count__") == nw.col("__max_count__"))
                .sort([uid_col, "__first_idx__"])
                .group_by([uid_col])
                .agg(
                    nw.col(lat_col).first().alias(lat_col),
                    nw.col(lng_col).first().alias(lng_col),
                )
            )
            return best_per_user.select([uid_col, lat_col, lng_col])
        else:
            # Single-user case: pick the (lat, lng) with the highest count,
            # then lowest first-occurrence index for tie-breaking.
            best = visit_counts.sort(["__count__", "__first_idx__"], descending=[True, False]).rows(named=True)[0]
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
