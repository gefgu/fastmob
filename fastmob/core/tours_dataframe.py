"""Tours — one row per tour (a maximal run of consecutive trips that starts
and ends at the same location).

Requires each trip's ``origin_staypoint_id``/``destination_staypoint_id`` to
resolve to a ``location_id`` via an already-clustered `Staypoints` table
(see `Staypoints.generate_user_locations`). Narwhals performs the joins; the
stateful per-user tour scan runs in the Rust hierarchy kernel.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import tours_from_trips
from fastmob.utils._common import (
    _as_arrow,
    _factorize_uids_uint32,
    _list_column_from_offsets,
    _narwhals_safe_value,
    _values_to_list,
)

from .base import BaseDataFrame

_REQUIRED_COLUMNS = ["tour_id", "started_at", "finished_at"]
_NULL_I64 = -(2**63)


class Tours(BaseDataFrame):
    """One row per tour (a sequence of trips returning to its starting location).

    ``.df`` columns: ``tour_id``, ``started_at``, ``finished_at``,
    ``location_id`` (the anchor location the tour starts and ends at),
    ``journey`` (list of the trip ids making up the tour), plus ``uid_col``
    when present.

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
                raise ValueError(f"Tours is missing required columns: {missing}")

    @staticmethod
    def from_trips(trips: Any, staypoints_with_location: Any) -> Tours:
        """Group consecutive trips into tours.

        See :meth:`fastmob.core.trips_dataframe.Trips.generate_tours`.

        Parameters
        ----------
        trips : Trips
            Trip summaries to scan chronologically.
        staypoints_with_location : Staypoints
            Staypoints carrying ``staypoint_id`` and ``location_id`` columns.

        Returns
        -------
        Tours
            Round-trip summaries with an anchor location and journey IDs.

        Raises
        ------
        ValueError
            If staypoint location assignments are unavailable.

        Examples
        --------
        >>> tours = Tours.from_trips(trips, staypoints_with_location)  # doctest: +SKIP
        """
        sp_nw = nw.from_native(staypoints_with_location.df, eager_only=True)
        if "location_id" not in sp_nw.columns:
            raise ValueError(
                "Tours.from_trips requires staypoints_with_location to have a 'location_id' column; "
                "run Staypoints.generate_user_locations() first"
            )
        if "staypoint_id" not in sp_nw.columns:
            raise ValueError("Tours.from_trips requires staypoints_with_location to have a 'staypoint_id' column")

        uid_col = trips.uid_col
        group_key = uid_col if uid_col else "__uid__"

        tl_nw = nw.from_native(trips.df, eager_only=True)
        if not uid_col:
            tl_nw = tl_nw.with_columns(nw.lit(0, dtype=nw.Int64).alias(group_key))

        sp_lookup = sp_nw.select(
            nw.col("staypoint_id").cast(nw.Int64),
            nw.col("location_id").cast(nw.Int64),
        )
        origin_lookup = sp_lookup.rename(
            {"staypoint_id": "__origin_staypoint_key__", "location_id": "__origin_location_id__"}
        )
        destination_lookup = sp_lookup.rename(
            {"staypoint_id": "__destination_staypoint_key__", "location_id": "__destination_location_id__"}
        )
        tl_nw = tl_nw.with_columns(
            nw.col("origin_staypoint_id").fill_null(_NULL_I64).cast(nw.Int64).alias("__origin_staypoint_key__"),
            nw.col("destination_staypoint_id")
            .fill_null(_NULL_I64)
            .cast(nw.Int64)
            .alias("__destination_staypoint_key__"),
        )
        tl_nw = tl_nw.join(origin_lookup, on="__origin_staypoint_key__", how="left")
        tl_nw = tl_nw.join(destination_lookup, on="__destination_staypoint_key__", how="left")
        tl_nw = tl_nw.sort([group_key, "started_at"]).with_columns(
            nw.col("trip_id").cast(nw.Int64),
            nw.col("started_at").cast(nw.Datetime("us")).dt.timestamp("us").cast(nw.Int64).alias("__started_at_us__"),
            nw.col("finished_at").cast(nw.Datetime("us")).dt.timestamp("us").cast(nw.Int64).alias("__finished_at_us__"),
            nw.col("__origin_location_id__").fill_null(_NULL_I64).cast(nw.Int64),
            nw.col("__destination_location_id__").fill_null(_NULL_I64).cast(nw.Int64),
        )
        if uid_col:
            uid_codes, _num_groups = _factorize_uids_uint32(tl_nw, group_key, sort=False)
            tl_nw = tl_nw.with_columns(uid_codes)
            uid_code_col = "__fastmob_uid_codes__"
            code_to_uid = dict(
                zip(
                    tl_nw.get_column(uid_code_col).to_list(),
                    tl_nw.get_column(group_key).to_list(),
                )
            )
        else:
            uid_code_col = "__fastmob_uid_codes__"
            tl_nw = tl_nw.with_columns(nw.lit(0, dtype=nw.UInt32).alias(uid_code_col))
            code_to_uid = {}

        (
            out_uid_codes,
            out_started_at_us,
            out_finished_at_us,
            out_location_id,
            journey_offsets,
            flat_journey,
        ) = tours_from_trips(
            tl_nw.get_column(uid_code_col).to_arrow(),
            tl_nw.get_column("trip_id").to_arrow(),
            tl_nw.get_column("__started_at_us__").to_arrow(),
            tl_nw.get_column("__finished_at_us__").to_arrow(),
            tl_nw.get_column("__origin_location_id__").to_arrow(),
            tl_nw.get_column("__destination_location_id__").to_arrow(),
        )

        uid_codes_list = _values_to_list(out_uid_codes)
        out_dict: dict[str, Any] = {
            "tour_id": list(range(len(uid_codes_list))),
            "__started_at_us__": _narwhals_safe_value(_as_arrow(out_started_at_us)),
            "__finished_at_us__": _narwhals_safe_value(_as_arrow(out_finished_at_us)),
            "location_id": _narwhals_safe_value(_as_arrow(out_location_id)),
            "journey": _list_column_from_offsets(flat_journey, journey_offsets),
        }
        if uid_col:
            out_dict[uid_col] = [code_to_uid[code] for code in uid_codes_list]

        out = nw.from_dict(out_dict, backend=tl_nw.implementation).with_columns(
            nw.col("__started_at_us__").cast(nw.Datetime("us")).alias("started_at"),
            nw.col("__finished_at_us__").cast(nw.Datetime("us")).alias("finished_at"),
        )
        out = out.drop("__started_at_us__", "__finished_at_us__")
        column_order = ([uid_col] if uid_col else []) + [
            "tour_id",
            "started_at",
            "finished_at",
            "location_id",
            "journey",
        ]
        return Tours(out.select(column_order).to_native(), uid_col=uid_col)
