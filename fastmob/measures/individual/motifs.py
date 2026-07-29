"""
Mobility motif classification.

Delegates the compute-heavy per-user-per-day graph construction and
canonicalization to the Rust kernel ``_core.compute_daily_motifs``, which
parallelises across users with Rayon.  The Python wrapper handles only column
extraction and result assembly.

Two things keep this measure off the sorting/loop paths a naive port would
reach for:

- **No full-dataframe sort.**  Motifs are order-dependent (a day's visit
  sequence matters), but that does not require physically sorting the
  trajectory dataframe: ``_build_time_ordered_user_ranges`` builds a
  time-ordered row-index permutation from a skinny (uid_code, timestamp)
  sort instead, the same pattern ``jump_lengths``/``radius_of_gyration``
  use for their default (non-``presorted``) path. The Rust kernel reads
  each row through that permutation rather than assuming a physically
  reordered input. Output row order is therefore whatever Rayon's
  (deterministic, but not otherwise meaningful) per-user-range processing
  order produces -- it is intentionally not re-sorted at the end.
- **No string node identity.**  Location+purpose identity and the "is this
  a HOME node" check are the two things the Rust kernel needs about each
  visit besides its hour/day/duration, and neither needs a string:
  ``location_id`` and ``purpose`` are dictionary-encoded *separately* into
  dense ``UInt64`` codes and combined arithmetically
  (``location_code * num_purposes + purpose_code``) into a single node
  code, so no ``"location_HOME"``-style string column is ever built or
  hashed. Every dense numeric/boolean column is handed to Rust as an Arrow
  array per the project's Arrow-only binding convention (see CLAUDE.md,
  "Arrow-only Rust bindings for dense numeric arrays") instead of a Python
  list PyO3 has to walk element-by-element.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob import _core
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    DURATION_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _extract_timestamps_ms,
    _factorize_uids_uint64,
    _narwhals_safe_value,
    _pick_existing_column,
)

# Only used to pick the NumPy-vs-Arrow extraction for the timestamps series
# fed into _build_time_ordered_user_ranges; no measure-specific ops needed.
_TIME_DISPATCHER = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _strip_time_zone(df: nw.DataFrame, column: str) -> nw.DataFrame:
    """Drop a tz-aware datetime column's time zone so it can cast to naive."""
    dtype = df.schema[column]
    if isinstance(dtype, nw.Datetime) and dtype.time_zone is not None:
        return df.with_columns(nw.col(column).dt.replace_time_zone(None))
    return df


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Vectorized bit-count for a uint64 NumPy array.

    Plain ``int.bit_count()`` only works one Python int at a time; NumPy's
    own ``np.bitwise_count`` needs NumPy>=2, which this project does not
    require. This is the standard SWAR bit-trick, entirely vectorized.
    """
    values = values.astype(np.uint64)
    values = values - ((values >> np.uint64(1)) & np.uint64(0x5555555555555555))
    values = (values & np.uint64(0x3333333333333333)) + ((values >> np.uint64(2)) & np.uint64(0x3333333333333333))
    values = (values + (values >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (values * np.uint64(0x0101010101010101)) >> np.uint64(56)


def discover_daily_motifs_from_agents(
    df: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    purpose_col: str | None = None,
    timestamp_col: str | None = None,
    end_timestamp_col: str | None = None,
    duration_col: str | None = None,
) -> tuple[Any, Any]:
    """Discover daily mobility motifs for all agents in a dataset.

    For each user-day, builds a directed mobility graph (with primary-home
    forced to node 0) and returns its canonical motif ID.  Graph construction
    and canonicalization run in Rust; users are processed in parallel via Rayon.

    Parameters
    ----------
    df : DataFrame-like
        Input visits DataFrame (any backend).  Must contain at minimum
        user_id, location_id, purpose, start_timestamp, and end_timestamp
        columns (or their auto-detected equivalents).
    user_id_col : str or None, optional
        User identifier column.  Auto-detected if None.
    location_id_col : str or None, optional
        Location identifier column.  Auto-detected if None.
    purpose_col : str or None, optional
        Activity purpose column (``"HOME"``, ``"WORK"``, etc.).
        Defaults to ``"purpose"`` if found.
    timestamp_col : str or None, optional
        Start-time column.  Defaults to ``"start_timestamp"`` if found.
    end_timestamp_col : str or None, optional
        End-time column.  Defaults to ``"end_timestamp"`` if found.
    duration_col : str or None, optional
        Duration column used for primary-home selection.  Defaults to
        ``"duration_minutes"`` if found.

    Returns
    -------
    tuple[pandas.DataFrame or polars.DataFrame, pandas.DataFrame or polars.DataFrame]
        ``(daily_motifs_df, motif_distribution_df)`` in the same backend as
        the input ``df``.

        *daily_motifs_df*: one row per user-day with columns
        ``[user_id_col, "date", "motif_id", "num_nodes", "num_edges"]``.
        Row order is not meaningful (not sorted by user or date).

        *motif_distribution_df*: one row per distinct motif with columns
        ``["motif_id", "count", "percentage"]``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "agent_id": ["u1", "u1", "u1"],
    ...     "location_id": ["home", "work", "home"],
    ...     "purpose": ["HOME", "WORK", "HOME"],
    ...     "start_timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 09:00", "2020-01-01 18:00"]),
    ...     "end_timestamp": pd.to_datetime(["2020-01-01 08:00", "2020-01-01 17:00", "2020-01-01 23:00"]),
    ... })
    >>> daily_df, dist_df = discover_daily_motifs_from_agents(df)
    """
    import pyarrow as pa

    # End-timestamp candidates — motifs-specific, not in _common.py
    _ETS_CANDIDATES = ["end_timestamp", "end_time"]

    nw_df = nw.from_native(df, eager_only=True)
    backend = nw_df.implementation
    cols = nw_df.columns

    # Column auto-detection
    if user_id_col is None:
        user_id_col = _pick_existing_column(cols, USER_ID_CANDIDATES) or "agent_id"
    user_id_dtype = nw_df.schema[user_id_col]
    if location_id_col is None:
        location_id_col = _pick_existing_column(cols, LOCATION_CANDIDATES) or "location_id"
    if purpose_col is None:
        purpose_col = _pick_existing_column(cols, PURPOSE_CANDIDATES) or "purpose"
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(cols, TIMESTAMP_CANDIDATES) or "start_timestamp"
    if end_timestamp_col is None:
        end_timestamp_col = _pick_existing_column(cols, _ETS_CANDIDATES) or "end_timestamp"
    if duration_col is None:
        duration_col = _pick_existing_column(cols, DURATION_CANDIDATES)

    # Rename columns to canonical internal names
    rename_map: dict[str, str] = {}
    if purpose_col != "purpose":
        rename_map[purpose_col] = "purpose"
    if timestamp_col != "start_timestamp":
        rename_map[timestamp_col] = "start_timestamp"
    if end_timestamp_col != "end_timestamp":
        rename_map[end_timestamp_col] = "end_timestamp"
    if duration_col and duration_col != "duration_minutes":
        rename_map[duration_col] = "duration_minutes"

    work_df = nw_df.rename(rename_map)

    # Tz-aware sources (e.g. Brightkite's UTC-stamped check-ins) can't cast
    # straight to a naive Datetime dtype below; drop the zone first.
    work_df = _strip_time_zone(work_df, "start_timestamp")
    work_df = _strip_time_zone(work_df, "end_timestamp")

    # Build derived columns needed by the Rust kernel:
    #   is_home     = purpose == "HOME"
    #   start_hour  = hour of start_timestamp
    #   end_hour    = hour of end_timestamp
    #   date_id     = days since Unix epoch (Int64) — computed via truncate-to-day
    #                 then divide epoch-microseconds by 86400*1e6
    # (node identity itself is built below via separate location/purpose
    # factorization, not a string column here.)
    work_df = work_df.with_columns(
        nw.col("start_timestamp").cast(nw.Datetime("us")).alias("start_timestamp"),
        nw.col("end_timestamp").cast(nw.Datetime("us")).alias("end_timestamp"),
    ).with_columns(
        (nw.col("purpose") == nw.lit("HOME")).alias("is_home"),
        nw.col("start_timestamp").dt.hour().cast(nw.UInt64).alias("start_hour"),
        nw.col("end_timestamp").dt.hour().cast(nw.UInt64).alias("end_hour"),
        (nw.col("start_timestamp").dt.truncate("1d").cast(nw.Int64) // (86400 * 1_000_000)).alias("date_id"),
    )

    # Time-ordered per-user row-index permutation instead of physically
    # sorting the dataframe: this is the same "skinny index sort" pattern
    # jump_lengths/radius_of_gyration use for their default path (see
    # CLAUDE.md). The Rust kernel reads rows through `indices`.
    timestamps = _extract_timestamps_ms(work_df, "start_timestamp")
    timestamps_data = _TIME_DISPATCHER.get_ops(work_df)["extract_data"](timestamps)
    user_id_labels, indices, ends = _build_time_ordered_user_ranges(
        work_df, user_id_col, "start_timestamp", timestamps_data
    )

    # Dictionary-encode location and purpose *separately* into dense integer
    # codes and combine them arithmetically into a node code, instead of
    # concatenating "location_id + '_' + purpose" into a string column and
    # factorizing that. purpose has tiny cardinality, so this is effectively
    # "factorize location_id" plus a cheap multiply-add — no string
    # materialization or string hashing anywhere.
    location_codes, num_locations = _factorize_uids_uint64(work_df, location_id_col, sort=False)
    purpose_codes, num_purposes = _factorize_uids_uint64(work_df, "purpose", sort=False)
    location_codes_np = location_codes.to_numpy()
    purpose_codes_np = purpose_codes.to_numpy()
    node_codes_np = location_codes_np * np.uint64(num_purposes) + purpose_codes_np
    num_codes = int(num_locations) * int(num_purposes)

    is_home_np = np.asarray(work_df.get_column("is_home").to_numpy(), dtype=bool)
    is_home_by_code = np.zeros(num_codes, dtype=bool)
    is_home_by_code[node_codes_np] = is_home_np

    if "duration_minutes" in work_df.columns:
        durations_arrow = work_df.get_column("duration_minutes").cast(nw.Float64).to_arrow()
    else:
        durations_arrow = pa.nulls(len(work_df), type=pa.float64())

    # Call the Rust kernel: every measure-data argument is a dense Arrow
    # array (Arrow-only binding convention); `indices`/`ends` stay plain
    # NumPy, since they are index metadata, not measure data.
    raw_user_idx_out, raw_date_ids_out, raw_motif_ids_out = _core.compute_daily_motifs(
        pa.array(node_codes_np, type=pa.uint64()),
        work_df.get_column("is_home").to_arrow(),
        work_df.get_column("start_hour").to_arrow(),
        work_df.get_column("end_hour").to_arrow(),
        work_df.get_column("date_id").to_arrow(),
        durations_arrow,
        indices,
        ends,
        pa.array(is_home_by_code),
    )

    date_ids_out = _arrow_result_values(raw_date_ids_out)
    motif_ids_out = _arrow_result_values(raw_motif_ids_out)

    if len(motif_ids_out) == 0:
        empty_daily = nw.from_dict(
            {
                user_id_col: [],
                "date": [],
                "motif_id": [],
                "num_nodes": [],
                "num_edges": [],
            },
            backend=backend,
        ).with_columns(nw.col(user_id_col).cast(user_id_dtype))
        empty_dist = nw.from_dict(
            {"motif_id": [], "count": [], "percentage": []},
            backend=backend,
        )
        return empty_daily.to_native(), empty_dist.to_native()

    # Parse num_nodes and num_edges from packed motif IDs, vectorized.
    # motif_id format: (n_nodes << 36) | adjacency_bits, or -1.
    # motif_ids_out may be a pyarrow.Array or an arro3.core.Array (its
    # to_numpy() takes no zero_copy_only kwarg), so go through __array__.
    motif_ids_np = np.asarray(motif_ids_out)
    valid = motif_ids_np != -1
    mask36 = (1 << 36) - 1
    num_nodes_arr = np.where(valid, motif_ids_np >> 36, -1)
    adjacency_arr = np.where(valid, motif_ids_np & mask36, 0).astype(np.uint64)
    num_edges_arr = np.where(valid, _popcount64(adjacency_arr).astype(np.int64), -1)

    # Map each output row's user_ends position back to its native-dtype
    # user id label (gathered once above, never round-tripped through Rust
    # as a string).  This is a plain Python list rather than the shared
    # ``_take_uid_values`` vectorized-numpy-take helper on purpose: that
    # helper builds an intermediate ``dtype=object`` NumPy array, which
    # Polars represents as an ``Object`` column that the explicit
    # ``.cast(user_id_dtype)`` below cannot cast back from. A plain list of
    # native-dtype Python values lets every backend infer the correct dtype
    # directly, and this loop runs over output rows (one per user-day, not
    # per input row), so it is not a hot path.
    user_ids_out = [user_id_labels[idx] for idx in raw_user_idx_out]

    # Convert date_ids_out (days since epoch) back to Datetime(us)
    # date_id * 86400 * 1_000_000 microseconds → Datetime('us').  Row order
    # is intentionally left as Rust produced it (deterministic, but not
    # sorted by user/date) rather than paying for another full sort here.
    daily_nw = (
        nw.from_dict(
            {
                user_id_col: user_ids_out,
                "date_id_raw": _narwhals_safe_value(date_ids_out),
                "motif_id": _narwhals_safe_value(motif_ids_out),
                "num_nodes": num_nodes_arr,
                "num_edges": num_edges_arr,
            },
            backend=backend,
        )
        .with_columns(
            (nw.col("date_id_raw").cast(nw.Int64) * (86400 * 1_000_000)).cast(nw.Datetime("us")).alias("date"),
            nw.col(user_id_col).cast(user_id_dtype),
        )
        .drop("date_id_raw")
    )

    daily_df = daily_nw.to_native()
    dist_df = compute_daily_motifs_distribution(daily_df)

    return daily_df, dist_df


def compute_daily_motifs_distribution(
    daily_motifs_df: Any,
    motif_id_col: str = "motif_id",
) -> Any:
    """Compute the distribution of motifs from a daily motifs DataFrame.

    Parameters
    ----------
    daily_motifs_df:
        DataFrame containing at least a column with motif IDs (e.g. output
        from ``discover_daily_motifs_from_agents``).
    motif_id_col:
        Column name for the motif ID.  Default ``"motif_id"``.

    Returns
    -------
    DataFrame
        One row per distinct motif with columns
        ``["motif_id", "count", "percentage"]``.
    """
    nw_df = nw.from_native(daily_motifs_df, eager_only=True)
    total_rows = len(nw_df)
    dist_nw = (
        nw_df.group_by(motif_id_col)
        .agg(nw.len().alias("count"))
        .sort(motif_id_col)
        .with_columns((nw.col("count") / total_rows * 100).alias("percentage"))
    )
    return dist_nw.to_native()
