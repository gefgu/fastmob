from __future__ import annotations

import math
from typing import Any, Callable

import narwhals as nw
import numpy as np

from fastmob._core import (
    SegmentConfig,
    segment_trajectory_indexed,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _as_arrow,
    _build_indexed_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps,
    _factorize_uids_uint32,
)

# nw.col(...).dt.truncate() bucket-length strings for each TemporalSplitter mode.
_TEMPORAL_TRUNCATE_UNITS: dict[str, str] = {
    "hour": "1h",
    "day": "1d",
    "month": "1mo",
    "year": "1y",
}


def _assign_segment_column(df: nw.DataFrame, segment_ids: Any) -> Any:
    """Add a ``segment_id`` column to ``df`` and return the native (backend) dataframe.

    @usedBy `segment()`. Both the Arrow and NumPy routes share this single
    implementation: Narwhals' ``new_series`` accepts both plain NumPy arrays
    and PyArrow arrays, so there is no backend-specific branching needed
    here — unlike `simplify`/`filter`'s row-filtering logic, which does
    differ by backend because it filters rows rather than adding a column.
    """
    return df.with_columns(nw.new_series("segment_id", segment_ids, backend=df.implementation)).to_native()


# Bare extractor kept only for the timestamps_data extraction below, which
# also feeds _build_indexed_user_ranges and so must keep matching
# whatever backend that helper's own uid-code extraction picks internally
# (see waiting_times.py/mean_square_displacement.py for why that pairing
# can't be forced to Arrow independently of the index-metadata builder).
_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _prepare_angle_change(min_angle: float = 45.0, min_speed_kmh: float = 0.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for AngleChange segmentation.

    @usedBy `segment()` via `SEGMENT_METHODS["angle_change"]`.
    """
    return "angle_change", {"min_angle_deg": min_angle, "angle_min_speed_kmh": min_speed_kmh}


def _prepare_observation_gap(gap_s: float = 3600.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for ObservationGap segmentation.

    @usedBy `segment()` via `SEGMENT_METHODS["observation_gap"]`.
    """
    return "observation_gap", {"gap_s": gap_s}


def _prepare_speed(
    speed_kmh: float = 0.0,
    max_speed_kmh: float = math.inf,
    duration_s: float = 300.0,
    **_: Any,
) -> tuple[str, dict]:
    """Build Rust config params for Speed segmentation.

    @usedBy `segment()` via `SEGMENT_METHODS["speed"]`.
    """
    return "speed", {
        "speed_min_kmh": speed_kmh,
        "speed_max_kmh": max_speed_kmh,
        "duration_s": duration_s,
    }


def _prepare_stop(
    stop_radius_km: float = 0.2,
    minutes_for_a_stop: float = 20.0,
    no_data_for_minutes: float = 1e12,
    min_speed_kmh: float | None = None,
    **_: Any,
) -> tuple[str, dict]:
    """Build Rust config params for Stop segmentation.

    @usedBy `segment()` via `SEGMENT_METHODS["stop"]`.
    """
    return "stop", {
        "stop_radius_km": stop_radius_km,
        "stop_minutes_for_a_stop": minutes_for_a_stop,
        "stop_no_data_for_minutes": no_data_for_minutes,
        "stop_min_speed_kmh": min_speed_kmh if min_speed_kmh is not None else math.inf,
    }


def _prepare_value_change(col_name: str | None = None, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for ValueChange segmentation.

    Unlike the other ``_prepare_*`` functions, this one's returned params
    dict carries a private ``__bucket_col__`` key (popped out in `segment()`
    before constructing `SegmentConfig`) naming the arbitrary column whose
    consecutive-value changes should be watched; ``segment()`` factorizes
    that column into bucket ids before dispatching to the Rust
    ``"value_change"`` kernel.

    @usedBy `segment()` via `SEGMENT_METHODS["value_change"]`.
    """
    if col_name is None:
        raise ValueError("segment(method='value_change') requires a 'col_name' keyword argument")
    return "value_change", {"__bucket_col__": col_name}


def _prepare_temporal(mode: str = "day", **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Temporal segmentation.

    Temporal is not a distinct Rust method: it resolves to the same
    ``"value_change"`` kernel ValueChange uses, with a truncated-datetime
    bucket-id array built in `segment()` standing in for an arbitrary
    column's values (see `_TEMPORAL_TRUNCATE_UNITS`). Like
    `_prepare_value_change`, this returns a private ``__temporal_mode__`` key
    popped out in `segment()` before constructing `SegmentConfig`.

    @usedBy `segment()` via `SEGMENT_METHODS["temporal"]`.
    """
    if mode not in _TEMPORAL_TRUNCATE_UNITS:
        raise ValueError(f"unknown temporal mode: {mode!r}; choose from {sorted(_TEMPORAL_TRUNCATE_UNITS)}")
    return "value_change", {"__temporal_mode__": mode}


SEGMENT_METHODS: dict[str, Callable[..., tuple[str, dict]]] = {
    "angle_change": _prepare_angle_change,
    "observation_gap": _prepare_observation_gap,
    "speed": _prepare_speed,
    "stop": _prepare_stop,
    "value_change": _prepare_value_change,
    "temporal": _prepare_temporal,
}


def _build_bucket_ids(df: nw.DataFrame, datetime_col: str, params: dict) -> np.ndarray | None:
    """Return an int64 bucket-id array for the ``value_change``/``temporal`` methods, or None.

    Pops the private ``__bucket_col__``/``__temporal_mode__`` key out of
    ``params`` in place (so the remaining dict is safe to pass straight into
    `SegmentConfig`), and returns bucket ids aligned 1:1 with ``df``'s row
    order (matching the row order `segment()`'s ``lats_data``/``lngs_data``
    extraction and index-building use).

    ``value_change`` factorizes an arbitrary named column via
    `_factorize_uids_uint32` (a generic dense-code factorizer despite its
    "uid" name — see that helper's docstring). ``temporal`` truncates the
    datetime column (`nw.col(...).dt.truncate()`) to the requested bucket
    length and uses the truncated Unix-ms timestamp directly as the bucket
    id: equal truncated timestamps only need to compare equal, not be
    contiguous integers.

    ``fill_null(0)`` before the final ``Int64`` cast is required for a null
    datetime value: pandas' non-nullable numpy int64 cast raises on NaN/NaT
    (unlike Polars/Arrow's nullable Int64), so every backend needs a
    concrete placeholder here. The placeholder never affects grouping — a
    row with a null datetime already has a non-finite value in `times_data`
    and is excluded by `is_valid_segment_row` on the Rust side regardless of
    its bucket id.

    @usedBy `segment()`.
    """
    if "__bucket_col__" in params:
        bucket_col = params.pop("__bucket_col__")
        bucket_series, _ = _factorize_uids_uint32(df, bucket_col, sort=False)
        return bucket_series.to_numpy().astype(np.int64)

    if "__temporal_mode__" in params:
        mode = params.pop("__temporal_mode__")
        unit = _TEMPORAL_TRUNCATE_UNITS[mode]
        bucket_series = df.with_columns(
            nw.col(datetime_col)
            .dt.truncate(unit)
            .dt.timestamp("ms")
            .fill_null(0)
            .cast(nw.Int64)
            .alias("__fastmob_temporal_bucket__")
        ).get_column("__fastmob_temporal_bucket__")
        return bucket_series.to_numpy().astype(np.int64)

    return None


def segment(
    traj: Any,
    method: str = "value_change",
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    is_sorted: bool = False,
    **method_kwargs: Any,
) -> Any:
    """Partition a trajectory into segments using a named algorithm.

    Unlike `simplify`/`filter` (which drop rows), `segment` never changes
    row count: it adds an integer ``segment_id`` column aligned 1:1 with the
    input, restarting at ``0`` for each user. Every shipped method is ported
    from MovingPandas' ``trajectory_splitter.py`` module, adapted to
    fastmob's hard row-preservation contract — where MovingPandas' splitters
    drop short/parked/gap rows entirely (e.g. a ``min_length`` cutoff, or
    every row strictly inside a detected stop), fastmob folds those rows
    forward into a neighboring segment (or, for ``stop``, gives them their
    own bracketed segment) instead of dropping them. This is a deliberate,
    documented semantic difference, not a bug — see each method's Rust
    kernel docstring for the exact adaptation.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    method:
        Name of the segmentation algorithm to run. One of
        ``"angle_change"``, ``"observation_gap"``, ``"speed"``, ``"stop"``,
        ``"value_change"`` (default), or ``"temporal"``.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    is_sorted:
        Whether the trajectory is already sorted by user and time. Setting
        this to True can speed up processing but may lead to incorrect
        results if the data is not properly preprocessed.
    **method_kwargs:
        Method-specific parameters, forwarded to the matching
        ``SEGMENT_METHODS[method]`` preparer.

        - ``angle_change``: ``min_angle`` (default ``45.0``) — minimum
          bearing change, in degrees, between the running reference heading
          and a point's incoming heading required to start a new segment;
          ``min_speed_kmh`` (default ``0.0``) — minimum incoming speed, in
          km/h, for a point to be evaluated at all.
        - ``observation_gap``: ``gap_s`` (default ``3600.0``) — time gap, in
          seconds, above which a new segment starts.
        - ``speed``: ``speed_kmh`` (default ``0.0``) and ``max_speed_kmh``
          (default ``inf``) — a point is "moving" when its incoming speed,
          in km/h, falls in ``[speed_kmh, max_speed_kmh]``; ``duration_s``
          (default ``300.0``) — minimum duration, in seconds, a contiguous
          non-moving run must span to bracket a new segment (shorter
          non-moving runs are folded into the surrounding segment).
        - ``stop``: ``stop_radius_km`` (default ``0.2``), ``minutes_for_a_stop``
          (default ``20.0``), ``no_data_for_minutes`` (default ``1e12``),
          ``min_speed_kmh`` (default ``None``, disabling end-trimming) — same
          stop-detection parameters as `stay_locations`; every row inside a
          detected stop's window becomes its own segment, bracketed by the
          moving segments before and after it.
        - ``value_change``: ``col_name`` (required) — name of the column
          whose consecutive-value changes start a new segment.
        - ``temporal``: ``mode`` (default ``"day"``) — one of ``"hour"``,
          ``"day"``, ``"month"``, ``"year"``; truncates the datetime column
          to that bucket length and starts a new segment on every bucket
          change (delegates to the same kernel as ``value_change``).

    Returns
    -------
    DataFrame
        Input trajectory with an added ``segment_id`` column, in the same
        backend as input.

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
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng"]
    ... ]
    >>> from fastmob.preprocessing import segment
    >>> segmented = segment(df, method="observation_gap", gap_s=3600.0)
    >>> len(segmented) == len(df)
    True
    >>> "segment_id" in segmented.columns
    True

    References
    ----------
    - [GJH2020] Graser, A., Jankowski, P., & Heistermann, M. (2020).
      MovingPandas. <a href="https://github.com/movingpandas/movingpandas">https://github.com/movingpandas/movingpandas</a>
    """
    if method not in SEGMENT_METHODS:
        raise ValueError(f"unknown segment method: {method!r}; choose from {sorted(SEGMENT_METHODS)}")
    method_name, params = SEGMENT_METHODS[method](**method_kwargs)

    df = nw.from_native(traj, eager_only=True)
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

    bucket_ids = _build_bucket_ids(df, datetime_col, params)

    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    timestamps = _extract_timestamps(df, datetime_col)
    times_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps)

    config = SegmentConfig(method=method_name, **params)

    if is_sorted:
        _, sorted_indices, ends = _build_indexed_user_ranges(df, uid_col)
        raw_ids = segment_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config, bucket_ids)
    else:
        _, sorted_indices, ends = _build_indexed_user_ranges(
            df,
            uid_col,
            timestamps=timestamps,
        )
        raw_ids = segment_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config, bucket_ids)

    segment_ids = np.asarray(_as_arrow(raw_ids), dtype=np.uint32)
    result = _assign_segment_column(df, segment_ids)

    return result


segment.__module__ = "fastmob.preprocessing"
