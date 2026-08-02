"""Trips — one row per trip (a maximal run of consecutive triplegs separated
only by non-activity staypoints).

Requires `staypoints` to already carry an ``activity`` boolean column (see
`fastmob.preprocessing.create_activity_flag`): a trip boundary is defined as
a staypoint with ``activity=True``; everything between two activity
staypoints (including any short "waiting" staypoints along the way) forms
one trip -- matching trackintel's own `generate_trips` semantics.

Built by merging the per-tripleg/per-staypoint tables via Narwhals, then
handing primitive backend-native buffers to the Rust hierarchy kernel for
the stateful per-user scan.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import trips_from_timeline
from fastmob.utils._common import (
    _as_arrow,
    _factorize_uids_uint64,
    _list_column_from_offsets,
    _narwhals_safe_value,
    _null_sentinel_to_none,
    _values_to_list,
)

from .base import BaseDataFrame

_REQUIRED_COLUMNS = ["trip_id", "started_at", "finished_at"]
_NULL_I64 = -(2**63)


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
            nw.lit(1, dtype=nw.UInt8).alias("__kind_code__"),
        )
        sp_ts = sp_nw.select(
            *([nw.col(uid_col)] if uid_col else [nw.lit(0, dtype=nw.Int64).alias(group_key)]),
            nw.col("staypoint_id"),
            nw.col("activity"),
            nw.col(started_at_col).cast(nw.Datetime("us")).alias("started_at"),
            nw.col(finished_at_col).cast(nw.Datetime("us")).alias("finished_at"),
            nw.lit(0, dtype=nw.UInt8).alias("__kind_code__"),
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
        timeline = timeline.with_columns(
            nw.col("activity").fill_null(value=False).cast(nw.Boolean),
            nw.col("staypoint_id").fill_null(_NULL_I64).cast(nw.Int64),
            nw.col("tripleg_id").fill_null(_NULL_I64).cast(nw.Int64),
            nw.col("started_at").dt.timestamp("us").cast(nw.Int64).alias("__started_at_us__"),
            nw.col("finished_at").dt.timestamp("us").cast(nw.Int64).alias("__finished_at_us__"),
        )
        if uid_col:
            uid_codes, _num_groups = _factorize_uids_uint64(timeline, group_key, sort=False)
            timeline = timeline.with_columns(uid_codes)
            uid_code_col = "__fastmob_uid_codes__"
            code_to_uid = dict(
                zip(
                    timeline.get_column(uid_code_col).to_list(),
                    timeline.get_column(group_key).to_list(),
                )
            )
        else:
            uid_code_col = "__fastmob_uid_codes__"
            timeline = timeline.with_columns(nw.lit(0, dtype=nw.UInt64).alias(uid_code_col))
            code_to_uid = {}

        (
            out_uid_codes,
            out_started_at_us,
            out_finished_at_us,
            out_origin_staypoint_id,
            out_destination_staypoint_id,
            tripleg_offsets,
            flat_tripleg_ids,
        ) = trips_from_timeline(
            timeline.get_column(uid_code_col).to_arrow(),
            timeline.get_column("__kind_code__").to_arrow(),
            timeline.get_column("activity").to_arrow(),
            timeline.get_column("staypoint_id").to_arrow(),
            timeline.get_column("tripleg_id").to_arrow(),
            timeline.get_column("__started_at_us__").to_arrow(),
            timeline.get_column("__finished_at_us__").to_arrow(),
        )

        uid_codes_list = _values_to_list(out_uid_codes)
        origin_ids = _null_sentinel_to_none(_values_to_list(out_origin_staypoint_id), _NULL_I64)
        destination_ids = _null_sentinel_to_none(_values_to_list(out_destination_staypoint_id), _NULL_I64)
        tripleg_ids = _list_column_from_offsets(flat_tripleg_ids, tripleg_offsets)

        out_dict: dict[str, Any] = {
            "trip_id": list(range(len(uid_codes_list))),
            "__started_at_us__": _narwhals_safe_value(_as_arrow(out_started_at_us)),
            "__finished_at_us__": _narwhals_safe_value(_as_arrow(out_finished_at_us)),
            "origin_staypoint_id": origin_ids,
            "destination_staypoint_id": destination_ids,
            "tripleg_ids": tripleg_ids,
        }
        if uid_col:
            out_dict[uid_col] = [code_to_uid[code] for code in uid_codes_list]

        out = nw.from_dict(out_dict, backend=tl_nw.implementation).with_columns(
            nw.col("__started_at_us__").cast(nw.Datetime("us")).alias("started_at"),
            nw.col("__finished_at_us__").cast(nw.Datetime("us")).alias("finished_at"),
        )
        out = out.drop("__started_at_us__", "__finished_at_us__")
        column_order = ([uid_col] if uid_col else []) + [
            "trip_id",
            "started_at",
            "finished_at",
            "origin_staypoint_id",
            "destination_staypoint_id",
            "tripleg_ids",
        ]
        return Trips(out.select(column_order).to_native(), uid_col=uid_col)

    def generate_tours(self, staypoints_with_location: Any) -> Any:
        """Group consecutive trips into tours (round trips back to the same location).

        See :meth:`fastmob.core.tours_dataframe.Tours.from_trips`.
        """
        from .tours_dataframe import Tours

        return Tours.from_trips(self, staypoints_with_location)
