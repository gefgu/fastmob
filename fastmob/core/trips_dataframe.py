"""Trips — one row per trip (a maximal run of consecutive triplegs separated
only by non-activity staypoints).

Requires `staypoints` to already carry an ``activity`` boolean column (see
`fastmob.preprocessing.create_activity_flag`): a trip boundary is defined as
a staypoint with ``activity=True``; everything between two activity
staypoints (including any short "waiting" staypoints along the way) forms
one trip -- matching trackintel's own `generate_trips` semantics.

Built by merging the (much smaller than raw-GPS-fix-count) per-tripleg/
per-staypoint tables via Narwhals, then extracting the merged timeline to
NumPy for the scan itself (Narwhals has no direct "collect to list per
group" aggregation, and this is not a performance-critical path) --
`import pandas`/`import polars` stay forbidden inside `fastmob/` per
CLAUDE.md; NumPy is not restricted by that rule.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from .base import BaseDataFrame

_REQUIRED_COLUMNS = ["trip_id", "started_at", "finished_at"]


class Trips(BaseDataFrame):
    """One row per trip.

    ``.df`` columns: ``trip_id``, ``started_at``, ``finished_at``,
    ``origin_staypoint_id``, ``destination_staypoint_id``, ``tripleg_ids``
    (a list of the tripleg ids making up the trip), plus ``uid_col`` when
    present. ``origin_staypoint_id`` is null for a user's very first trip
    (no staypoint precedes it); ``destination_staypoint_id`` is null for a
    user's last trip if it never reaches another activity staypoint.

    Parameters
    ----------
    df : DataFrame-like
        Source data; any Narwhals-compatible eager backend.
    uid_col : str, optional
        User-ID column name.
    validate : bool, optional
        When True (default), check that required columns are present.
    """

    def __init__(self, df: Any, uid_col: str | None = None, validate: bool = True):
        super().__init__(df)
        self.uid_col = uid_col
        if validate:
            nw_df = nw.from_native(df, eager_only=True)
            missing = [col for col in _REQUIRED_COLUMNS if col not in nw_df.columns]
            if missing:
                raise ValueError(f"Trips is missing required columns: {missing}")

    @staticmethod
    def from_triplegs(triplegs: Any, staypoints: Any, gap_threshold_min: float = 15.0) -> Trips:
        """Derive trips from triplegs + activity-flagged staypoints.

        See :meth:`fastmob.core.triplegs_dataframe.Triplegs.generate_trips`.
        """
        del gap_threshold_min  # reserved; the current method needs only the activity flag

        sp_nw = nw.from_native(staypoints.df, eager_only=True)
        if "activity" not in sp_nw.columns:
            raise ValueError(
                "Trips.from_triplegs requires staypoints to have an 'activity' column; "
                "run Staypoints.create_activity_flag() first"
            )

        uid_col = triplegs.uid_col
        started_at_col = staypoints.started_at_col
        finished_at_col = staypoints.finished_at_col

        tl_nw = nw.from_native(triplegs.df, eager_only=True)
        n_sp = len(sp_nw)
        sp_nw = sp_nw if "staypoint_id" in sp_nw.columns else sp_nw.with_row_index("staypoint_id")

        group_key = uid_col if uid_col else "__uid__"
        # Cast to a common datetime precision before concatenating -- Triplegs'
        # started_at/finished_at and Staypoints' own can carry different
        # backend-native precisions (e.g. Polars microseconds vs. a
        # nanosecond column), which `nw.concat` rejects as a schema mismatch.
        tl_ts = tl_nw.select(
            *([nw.col(uid_col)] if uid_col else [nw.lit(0, dtype=nw.Int64).alias(group_key)]),
            nw.col("tripleg_id"),
            nw.col("started_at").cast(nw.Datetime("us")),
            nw.col("finished_at").cast(nw.Datetime("us")),
            nw.lit("tripleg").alias("__kind__"),
        )
        sp_ts = sp_nw.select(
            *([nw.col(uid_col)] if uid_col else [nw.lit(0, dtype=nw.Int64).alias(group_key)]),
            nw.col("staypoint_id"),
            nw.col("activity"),
            nw.col(started_at_col).cast(nw.Datetime("us")).alias("started_at"),
            nw.col(finished_at_col).cast(nw.Datetime("us")).alias("finished_at"),
            nw.lit("stop").alias("__kind__"),
        )
        if n_sp == 0:
            return Trips(
                nw.from_dict(
                    {
                        **({uid_col: []} if uid_col else {}),
                        "trip_id": [],
                        "started_at": [],
                        "finished_at": [],
                        "origin_staypoint_id": [],
                        "destination_staypoint_id": [],
                        "tripleg_ids": [],
                    },
                    backend=tl_nw.implementation,
                ).to_native(),
                uid_col=uid_col,
            )

        timeline = nw.concat([tl_ts, sp_ts], how="diagonal").sort([group_key, "started_at"])
        timeline = timeline.with_columns(nw.col("activity").fill_null(value=False))

        n = len(timeline)
        kind = timeline.get_column("__kind__").to_numpy()
        uid_values = timeline.get_column(group_key).to_numpy()
        activity = timeline.get_column("activity").to_numpy().astype(bool)
        staypoint_id = timeline.get_column("staypoint_id").to_numpy()
        tripleg_id = timeline.get_column("tripleg_id").to_numpy()
        started_at = timeline.get_column("started_at").to_numpy()
        finished_at = timeline.get_column("finished_at").to_numpy()

        is_activity_stop = (kind == "stop") & activity

        first_of_user = np.empty(n, dtype=bool)
        first_of_user[0] = True
        first_of_user[1:] = uid_values[1:] != uid_values[:-1]

        # Forward/backward-fill the most recent/next activity staypoint id
        # across every row (including tripleg rows), reset at each user
        # boundary -- pure NumPy since Narwhals has no cross-backend
        # "extract nullable column to array, ffill/bfill" primitive that's
        # simpler than just doing the scan directly here.
        origin_staypoint_id = np.full(n, np.nan)
        destination_staypoint_id = np.full(n, np.nan)
        last_seen = np.nan
        for i in range(n):
            if first_of_user[i]:
                last_seen = np.nan
            if is_activity_stop[i]:
                last_seen = staypoint_id[i]
            origin_staypoint_id[i] = last_seen
        next_seen = np.nan
        for i in range(n - 1, -1, -1):
            if i + 1 < n and first_of_user[i + 1]:
                next_seen = np.nan
            if is_activity_stop[i]:
                next_seen = staypoint_id[i]
            destination_staypoint_id[i] = next_seen

        boundary = is_activity_stop.astype(np.int64)
        cum = np.zeros(n, dtype=np.int64)
        running = 0
        for i in range(n):
            if first_of_user[i]:
                running = 0
            local_idx = running
            running += boundary[i]
            cum[i] = local_idx
        local_trip_idx = cum

        is_tripleg = kind == "tripleg"
        trips: dict[tuple[Any, int], dict[str, Any]] = {}
        order: list[tuple[Any, int]] = []
        for i in range(n):
            if not is_tripleg[i]:
                continue
            key = (uid_values[i], local_trip_idx[i])
            if key not in trips:
                trips[key] = {
                    "started_at": started_at[i],
                    "finished_at": finished_at[i],
                    "origin_staypoint_id": origin_staypoint_id[i],
                    "destination_staypoint_id": destination_staypoint_id[i],
                    "tripleg_ids": [],
                }
                order.append(key)
            entry = trips[key]
            entry["started_at"] = min(entry["started_at"], started_at[i])
            entry["finished_at"] = max(entry["finished_at"], finished_at[i])
            entry["tripleg_ids"].append(tripleg_id[i])

        # Build fixed-dtype NumPy arrays rather than plain Python lists of
        # NumPy scalars -- some backends (e.g. Polars) can't infer a schema
        # from a list of individual `numpy.datetime64`/`float64` objects.
        out_dict: dict[str, Any] = {
            "trip_id": np.arange(len(order), dtype=np.int64),
            "started_at": np.array([trips[k]["started_at"] for k in order], dtype="datetime64[us]"),
            "finished_at": np.array([trips[k]["finished_at"] for k in order], dtype="datetime64[us]"),
            "origin_staypoint_id": np.array([trips[k]["origin_staypoint_id"] for k in order], dtype=np.float64),
            "destination_staypoint_id": np.array(
                [trips[k]["destination_staypoint_id"] for k in order], dtype=np.float64
            ),
            "tripleg_ids": [trips[k]["tripleg_ids"] for k in order],
        }
        if uid_col:
            out_dict[uid_col] = [k[0] for k in order]

        return Trips(nw.from_dict(out_dict, backend=tl_nw.implementation).to_native(), uid_col=uid_col)

    def generate_tours(self, staypoints_with_location: Any) -> Any:
        """Group consecutive trips into tours (round trips back to the same location).

        See :meth:`fastmob.core.tours_dataframe.Tours.from_trips`.
        """
        from .tours_dataframe import Tours

        return Tours.from_trips(self, staypoints_with_location)
