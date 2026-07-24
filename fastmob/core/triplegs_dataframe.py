"""Triplegs — movement segments between staypoints.

Built by composing two already-Rust-backed operations -- no new Rust kernel
is needed for this level:

1. `fastmob.preprocessing.segment(method="stop")` partitions each user's
   positionfixes into alternating stop-window and moving runs
   (``segment_id``), using the *same* stop-detection parameters
   `Positionfixes.generate_staypoints` used.
2. Each stop-window run is matched against the corresponding `Staypoints`
   row by an exact ``(uid, started_at)`` join (both come from the same
   underlying stop-detection algorithm and parameters, so their computed
   entry times line up exactly) and discarded; the remaining "moving" runs
   are the triplegs.
3. Each tripleg's length is the sum of consecutive-point Haversine distances
   between its own rows, computed by reusing
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

        # Discard stop-window segments: an exact (uid, started_at) match
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
            .with_columns(nw.col(staypoints.started_at_col).dt.timestamp("ms").alias("__started_at_ms__"))
            .select(([uid_col] if uid_col else []) + ["__started_at_ms__"])
        )
        anti_join_cols = ([uid_col] if uid_col else []) + ["__started_at_ms__"]
        triplegs_summary = summary.join(stop_keys, on=anti_join_cols, how="anti").drop("__started_at_ms__")

        length_by_group = _tripleg_lengths(seg_nw, uid_col, datetime_col, lat_col, lng_col)
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
    uid_col: str | None,
    datetime_col: str,
    lat_col: str,
    lng_col: str,
) -> nw.DataFrame:
    """Sum of consecutive-point Haversine distances per ``(uid, segment_id)`` group.

    Reuses `jump_lengths`'s existing batched, presorted Rust kernel: a single
    ``merge=True`` call over the whole (already uid/time-sorted) frame
    returns one flat array of per-user consecutive distances (length
    ``n_u - 1`` per user); this is reconstructed here into a full per-row
    "distance from previous row" NumPy array (first row of each user gets
    0.0), which then group-sums correctly per ``segment_id`` -- no per-row
    Python loop, no new Rust kernel.
    """
    from ..measures.individual import jump_lengths

    n = len(seg_nw)
    if n == 0:
        empty_cols = ([uid_col] if uid_col else []) + ["segment_id", "length_km"]
        return nw.from_dict({col: [] for col in empty_cols}, backend=seg_nw.implementation)

    if uid_col:
        uid_values = seg_nw.get_column(uid_col).to_numpy()
        first_mask = np.empty(n, dtype=bool)
        first_mask[0] = True
        first_mask[1:] = uid_values[1:] != uid_values[:-1]
    else:
        first_mask = np.zeros(n, dtype=bool)
        first_mask[0] = True

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
    per_row_distance[~first_mask] = np.asarray(flat_distances, dtype=np.float64)

    keyed = seg_nw.with_columns(nw.new_series("__distance_km__", per_row_distance, backend=seg_nw.implementation))
    group_cols = [uid_col, "segment_id"] if uid_col else ["segment_id"]
    return keyed.group_by(group_cols).agg(nw.col("__distance_km__").sum().alias("length_km"))
