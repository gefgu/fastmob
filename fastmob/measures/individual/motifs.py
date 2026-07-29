"""
Mobility motif classification.

Delegates the compute-heavy per-user-per-day graph construction and
canonicalization to the Rust kernel ``_core.compute_daily_motifs``, which
parallelises across users with Rayon.  The Python wrapper handles only column
extraction and result assembly.

Location+purpose identity and the "is this a HOME node" check are the two
things the Rust kernel needs about each visit besides its hour/day/duration,
and neither needs the actual string: ``unique_id`` (``location_id + "_" +
purpose``) is dictionary-encoded into dense ``UInt64`` node codes via
``_factorize_uids_uint64`` before crossing into Rust, and every dense
numeric/boolean column is handed over as an Arrow array per the project's
Arrow-only binding convention (see CLAUDE.md, "Arrow-only Rust bindings for
dense numeric arrays") instead of a Python list PyO3 has to walk
element-by-element.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob import _core
from fastmob.utils._common import (
    DURATION_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _arrow_result_values,
    _build_presorted_user_ends,
    _factorize_uids_uint64,
    _narwhals_safe_value,
    _pick_existing_column,
)


def _strip_time_zone(df: nw.DataFrame, column: str) -> nw.DataFrame:
    """Drop a tz-aware datetime column's time zone so it can cast to naive."""
    dtype = df.schema[column]
    if isinstance(dtype, nw.Datetime) and dtype.time_zone is not None:
        return df.with_columns(nw.col(column).dt.replace_time_zone(None))
    return df


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
    #   unique_id   = location_id + "_" + purpose (dictionary-encoded below)
    #   is_home     = purpose == "HOME"
    #   start_hour  = hour of start_timestamp
    #   end_hour    = hour of end_timestamp
    #   date_id     = days since Unix epoch (Int64) — computed via truncate-to-day
    #                 then divide epoch-microseconds by 86400*1e6
    work_df = work_df.with_columns(
        nw.col("start_timestamp").cast(nw.Datetime("us")).alias("start_timestamp"),
        nw.col("end_timestamp").cast(nw.Datetime("us")).alias("end_timestamp"),
        nw.col("purpose").cast(nw.String).alias("purpose"),
        nw.col(location_id_col).cast(nw.String).alias(location_id_col),
    ).with_columns(
        (nw.col(location_id_col) + nw.lit("_") + nw.col("purpose")).alias("unique_id"),
        (nw.col("purpose") == nw.lit("HOME")).alias("is_home"),
        nw.col("start_timestamp").dt.hour().cast(nw.UInt64).alias("start_hour"),
        nw.col("end_timestamp").dt.hour().cast(nw.UInt64).alias("end_hour"),
        (nw.col("start_timestamp").dt.truncate("1d").cast(nw.Int64) // (86400 * 1_000_000))
        .cast(nw.Int64)
        .alias("date_id"),
    )

    # Sort by [user_id, start_timestamp] so visits within each user are
    # in chronological order and users are contiguous in the flat arrays
    work_df = work_df.sort([user_id_col, "start_timestamp"])

    # Per-user contiguous ranges via the shared Rust-backed presorted-range
    # builder (see "Rule 1" in CLAUDE.md) instead of a hand-rolled Python
    # scan over every row.
    user_id_labels, user_ends = _build_presorted_user_ends(work_df, user_id_col)

    # Dictionary-encode location+purpose into dense node codes.  The Rust
    # kernel only ever compares two visits' locations for equality and asks
    # "is this a HOME node" — it never needs the string itself — so send a
    # UInt64 code per row plus a small per-code "is this HOME" lookup built
    # once, instead of a full N-length Vec<String>.
    node_code_series, num_codes = _factorize_uids_uint64(work_df, "unique_id", sort=False)
    node_codes_np = node_code_series.to_numpy()
    is_home_series = work_df.get_column("is_home")
    is_home_np = np.asarray(is_home_series.to_numpy(), dtype=bool)

    is_home_by_code = np.zeros(num_codes, dtype=bool)
    is_home_by_code[node_codes_np] = is_home_np

    if "duration_minutes" in work_df.columns:
        durations_arrow = work_df.get_column("duration_minutes").cast(nw.Float64).to_arrow()
    else:
        durations_arrow = pa.nulls(len(work_df), type=pa.float64())

    # Call the Rust kernel: every measure-data argument is a dense Arrow
    # array (Arrow-only binding convention); only the per-user boundary
    # array stays plain NumPy, since it is index metadata, not measure data.
    raw_user_idx_out, raw_date_ids_out, raw_motif_ids_out = _core.compute_daily_motifs(
        node_code_series.to_arrow(),
        is_home_series.to_arrow(),
        work_df.get_column("start_hour").to_arrow(),
        work_df.get_column("end_hour").to_arrow(),
        work_df.get_column("date_id").to_arrow(),
        durations_arrow,
        user_ends,
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

    # Parse num_nodes and num_edges from packed motif IDs in Python
    # motif_id format: (n_nodes << 36) | adjacency_bits, or -1
    num_nodes_list: list[int] = []
    num_edges_list: list[int] = []

    for mid in motif_ids_out.to_pylist():
        if mid == -1:
            num_nodes_list.append(-1)
            num_edges_list.append(-1)
        else:
            # 1. Extract n_nodes by shifting right 36 bits
            n_nodes = mid >> 36
            num_nodes_list.append(n_nodes)

            # 2. Mask the lower 36 bits to get the adjacency matrix integer
            mask = (1 << 36) - 1
            adjacency_matrix = mid & mask

            # 3. Count the active bits (edges) natively in Python 3.10+
            num_edges_list.append(adjacency_matrix.bit_count())

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
    # date_id * 86400 * 1_000_000 microseconds → Datetime('us')
    daily_nw = (
        nw.from_dict(
            {
                user_id_col: user_ids_out,
                "date_id_raw": _narwhals_safe_value(date_ids_out),
                "motif_id": _narwhals_safe_value(motif_ids_out),
                "num_nodes": num_nodes_list,
                "num_edges": num_edges_list,
            },
            backend=backend,
        )
        .with_columns(
            (nw.col("date_id_raw").cast(nw.Int64) * (86400 * 1_000_000)).cast(nw.Datetime("us")).alias("date"),
            nw.col(user_id_col).cast(user_id_dtype),
        )
        .drop("date_id_raw")
        .sort([user_id_col, "date"])
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
