"""Triplegs — movement segments between staypoints.

Built by composing two already-Rust-backed operations -- no new Rust kernel
is needed for this level:

1. `fastmob.preprocessing.segment(method="stop")` partitions each user's
   positionfixes into alternating stop-window and moving runs
   (``segment_id``), using the *same* stop-detection parameters
   `Positionfixes.generate_staypoints` used.
2. Each stop-window segment is identified by an exact ``(uid, started_at)``
   match against the corresponding `Staypoints` row (both come from the
   same underlying stop-detection algorithm and parameters, so their
   computed entry times line up exactly); the remaining "moving" segments
   are the triplegs. `segment(method="stop")` attributes a stop's own
   entry/leaving transition rows to the *stop's* segment, not the moving
   segment before/after it -- so each tripleg's true door-to-door span is
   recovered by borrowing the bracketing stop segments' own
   finished_at/started_at, and its length includes the boundary hop that
   would otherwise be misattributed to the (discarded) stop segment. See
   `_tripleg_lengths`'s docstring for the exact redirection rule.
3. Each tripleg's length is the sum of consecutive-point Haversine distances
   along its full door-to-door span, computed by reusing
   `fastmob.measures.individual.jump_lengths`'s existing batched Rust
   kernel (a single ``merge=True`` flat-array call), not a new kernel.

Not built in this pass: `origin_staypoint_id`/`destination_staypoint_id`
columns linking each tripleg back to its bracketing `Staypoints` rows.
`Trips` generation (a later hierarchy level) needs that link and should
derive it itself (e.g. a nearest-preceding/following-staypoint-by-time
lookup), rather than this level precomputing and carrying columns nothing
here yet consumes.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from .base import BaseDataFrame

_REQUIRED_COLUMNS = ["tripleg_id", "started_at", "finished_at", "length_km", "duration_s"]


class Triplegs(BaseDataFrame):
    """One row per tripleg (a movement segment between two staypoints).

    ``.df`` is the per-tripleg summary table: ``tripleg_id``, ``started_at``,
    ``finished_at``, ``length_km``, ``duration_s``, ``mean_speed_kmh``, plus
    ``uid_col`` when present. Reconstructing each tripleg's point-by-point
    geometry (e.g. for a LineString) is not built in this pass.

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
                raise ValueError(f"Triplegs is missing required columns: {missing}")

    def predict_transport_mode(self, method: str = "simple-coarse", categories: dict | None = None) -> Triplegs:
        """Classify each tripleg's transport mode from its average speed.

        See :func:`fastmob.preprocessing.predict_transport_mode`.
        """
        from ..preprocessing import predict_transport_mode

        return predict_transport_mode(self, method=method, categories=categories)

    def calculate_modal_split(
        self,
        freq: str | None = None,
        metric: str = "count",
        per_user: bool = False,
        normalize: bool = False,
    ) -> Any:
        """Aggregate this ``mode``-labeled table into a modal-split table.

        See :func:`fastmob.preprocessing.calculate_modal_split`.
        """
        from ..preprocessing import calculate_modal_split

        return calculate_modal_split(self, freq=freq, metric=metric, per_user=per_user, normalize=normalize)

    def generate_trips(self, staypoints: Any, gap_threshold_min: float = 15.0) -> Any:
        """Group consecutive triplegs into trips.

        See :meth:`fastmob.core.trips_dataframe.Trips.from_triplegs`.
        """
        from .trips_dataframe import Trips

        return Trips.from_triplegs(self, staypoints, gap_threshold_min=gap_threshold_min)

    @staticmethod
    def from_positionfixes(
        positionfixes: Any,
        staypoints: Any,
        gap_threshold_min: float = 15.0,
        **stop_kwargs: Any,
    ) -> Triplegs:
        """Derive triplegs from positionfixes + already-generated staypoints.

        See :meth:`fastmob.core.positionfixes_dataframe.Positionfixes.generate_triplegs`.
        """
        from ..preprocessing import segment

        del gap_threshold_min  # reserved for a future gap-based splitting method

        uid_col = positionfixes.uid_col
        datetime_col = positionfixes.datetime_col
        lat_col = positionfixes.lat_col
        lng_col = positionfixes.lng_col

        params = getattr(staypoints, "parameters", {}) or {}
        seg_kwargs: dict[str, Any] = {
            "stop_radius_km": params.get("spatial_radius_km", 0.2),
            "minutes_for_a_stop": params.get("minutes_for_a_stop", 20.0),
        }
        if "no_data_for_minutes" in params:
            seg_kwargs["no_data_for_minutes"] = params["no_data_for_minutes"]
        if "min_speed_kmh" in params:
            seg_kwargs["min_speed_kmh"] = params["min_speed_kmh"]
        seg_kwargs.update(stop_kwargs)

        segmented = segment(
            positionfixes.df,
            method="stop",
            datetime_col=datetime_col,
            lat_col=lat_col,
            lng_col=lng_col,
            uid_col=uid_col,
            **seg_kwargs,
        )
        seg_nw = nw.from_native(segmented, eager_only=True)

        sort_cols = ([uid_col] if uid_col else []) + [datetime_col]
        seg_nw = seg_nw.sort(sort_cols)

        group_cols = [uid_col, "segment_id"] if uid_col else ["segment_id"]
        summary = seg_nw.group_by(group_cols).agg(
            nw.col(datetime_col).min().alias("started_at"),
            nw.col(datetime_col).max().alias("finished_at"),
            nw.len().alias("n_rows"),
        )
        # group_by doesn't guarantee output row order; re-sort chronologically
        # per user before the shift/over below (segment_id increases with
        # time per user, since seg_nw was itself sorted by (uid, datetime)).
        sort_by_segment = [uid_col, "segment_id"] if uid_col else ["segment_id"]
        summary = summary.sort(sort_by_segment)

        # `segment(method="stop")` attributes each stop's entry/leaving
        # transition rows to the STOP's own segment, not the moving segment
        # before/after it (see this module's docstring) -- so a moving
        # segment's own min/max datetime is truncated, sometimes to a single
        # degenerate point (0 duration despite a real, nonzero length).
        # Recover each tripleg's true door-to-door span by borrowing the
        # bracketing stop segments' own finished_at/started_at instead.
        if uid_col:
            prev_finished = nw.col("finished_at").shift(1).over(uid_col)
            next_started = nw.col("started_at").shift(-1).over(uid_col)
        else:
            prev_finished = nw.col("finished_at").shift(1)
            next_started = nw.col("started_at").shift(-1)
        summary = summary.with_columns(
            prev_finished.alias("__prev_finished_at__"),
            next_started.alias("__next_started_at__"),
        )
        summary = summary.with_columns(
            nw.col("__prev_finished_at__").fill_null(nw.col("started_at")).alias("__bracketed_started_at__"),
            nw.col("__next_started_at__").fill_null(nw.col("finished_at")).alias("__bracketed_finished_at__"),
        )

        # Identify stop-window segments: an exact (uid, started_at) match
        # against the staypoints this same positionfixes/parameters produced.
        # Join on millisecond-timestamp integers rather than the raw datetime
        # columns -- different backends can carry different datetime
        # precisions (e.g. Polars microseconds vs. a nanosecond column),
        # which a direct datetime-typed join key rejects as a type mismatch.
        summary = summary.with_columns(nw.col("started_at").dt.timestamp("ms").alias("__started_at_ms__"))
        stay_nw = nw.from_native(staypoints.df, eager_only=True)
        stop_keys_cols = ([uid_col] if uid_col else []) + [staypoints.started_at_col]
        stop_keys = (
            stay_nw.select(stop_keys_cols)
            .with_columns(
                nw.col(staypoints.started_at_col).dt.timestamp("ms").alias("__started_at_ms__"),
                nw.lit(True).alias("__is_stop__"),
            )
            .select(([uid_col] if uid_col else []) + ["__started_at_ms__", "__is_stop__"])
        )
        join_cols_ms = ([uid_col] if uid_col else []) + ["__started_at_ms__"]
        summary = summary.join(stop_keys, on=join_cols_ms, how="left").with_columns(
            nw.col("__is_stop__").fill_null(value=False).cast(nw.Boolean)
        )

        length_by_group = _tripleg_lengths(seg_nw, summary, uid_col, datetime_col, lat_col, lng_col, group_cols)

        triplegs_summary = summary.filter(~nw.col("__is_stop__")).drop("__started_at_ms__", "__is_stop__")
        triplegs_summary = (
            triplegs_summary.drop("started_at", "finished_at")
            .rename({"__bracketed_started_at__": "started_at", "__bracketed_finished_at__": "finished_at"})
            .drop("__prev_finished_at__", "__next_started_at__")
        )

        triplegs_summary = triplegs_summary.join(length_by_group, on=group_cols, how="left")
        triplegs_summary = triplegs_summary.with_columns(nw.col("length_km").fill_null(0.0))

        triplegs_summary = triplegs_summary.with_columns(
            ((nw.col("finished_at").dt.timestamp("ms") - nw.col("started_at").dt.timestamp("ms")) / 1000.0).alias(
                "duration_s"
            )
        )
        triplegs_summary = triplegs_summary.with_columns(
            nw.when(nw.col("duration_s") > 0)
            .then(nw.col("length_km") / (nw.col("duration_s") / 3600.0))
            .otherwise(0.0)
            .alias("mean_speed_kmh")
        ).drop("n_rows")
        triplegs_summary = triplegs_summary.rename({"segment_id": "tripleg_id"})

        return Triplegs(triplegs_summary.to_native(), uid_col=uid_col)


def _tripleg_lengths(
    seg_nw: nw.DataFrame,
    summary_with_is_stop: nw.DataFrame,
    uid_col: str | None,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
    group_cols: list[str],
) -> nw.DataFrame:
    """Sum of consecutive-point Haversine distances per moving ``(uid, segment_id)`` group.

    Reuses `jump_lengths`'s existing batched, presorted Rust kernel: a single
    ``merge=True`` call over the whole (already uid/time-sorted) frame
    returns one flat array of per-user consecutive distances (length
    ``n_u - 1`` per user); this is reconstructed here into a full per-row
    "distance from previous row" NumPy array (first row of each user gets
    0.0).

    `segment(method="stop")` attributes a stop's own entry-transition row to
    the STOP's segment, not the moving segment before it (see this module's
    docstring) -- so the hop leading into a stop would otherwise be
    misattributed to the (discarded) stop segment and silently dropped from
    the tripleg's length. Each such hop is redirected here to the
    *preceding* segment (the moving one) before grouping.
    """
    from ..measures.individual import jump_lengths

    n = len(seg_nw)
    empty_cols = ([uid_col] if uid_col else []) + ["segment_id", "length_km"]
    if n == 0:
        return nw.from_dict({col: [] for col in empty_cols}, backend=seg_nw.implementation)

    is_stop_lookup = summary_with_is_stop.select([*group_cols, "__is_stop__"])
    seg_with_flag = seg_nw.join(is_stop_lookup, on=group_cols, how="left").with_columns(
        nw.col("__is_stop__").fill_null(value=False).cast(nw.Boolean)
    )

    if uid_col:
        uid_values = seg_with_flag.get_column(uid_col).to_numpy()
        first_of_user = np.empty(n, dtype=bool)
        first_of_user[0] = True
        first_of_user[1:] = uid_values[1:] != uid_values[:-1]
    else:
        uid_values = np.zeros(n, dtype=np.int64)
        first_of_user = np.zeros(n, dtype=bool)
        first_of_user[0] = True

    seg_ids = seg_with_flag.get_column("segment_id").to_numpy()
    is_stop_row = seg_with_flag.get_column("__is_stop__").to_numpy()

    first_of_segment = np.empty(n, dtype=bool)
    first_of_segment[0] = True
    first_of_segment[1:] = (seg_ids[1:] != seg_ids[:-1]) | first_of_user[1:]

    redirect_mask = first_of_segment & is_stop_row & ~first_of_user
    attribution_segment_id = seg_ids.copy()
    redirect_idx = np.flatnonzero(redirect_mask)
    attribution_segment_id[redirect_idx] = seg_ids[redirect_idx - 1]

    flat_distances = jump_lengths(
        seg_nw.to_native(),
        merge=True,
        presorted=True,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    flat_distances = flat_distances.to_numpy() if hasattr(flat_distances, "to_numpy") else np.asarray(flat_distances)

    per_row_distance = np.zeros(n, dtype=np.float64)
    per_row_distance[~first_of_user] = np.asarray(flat_distances, dtype=np.float64)

    lengths_dict: dict[str, Any] = {
        "segment_id": attribution_segment_id,
        "__distance_km__": per_row_distance,
    }
    if uid_col:
        lengths_dict["__uid__"] = uid_values
    lengths_nw = nw.from_dict(lengths_dict, backend=seg_nw.implementation)

    group_by_cols = ["__uid__", "segment_id"] if uid_col else ["segment_id"]
    grouped = lengths_nw.group_by(group_by_cols).agg(nw.col("__distance_km__").sum().alias("length_km"))
    if uid_col:
        grouped = grouped.rename({"__uid__": uid_col})
    return grouped
