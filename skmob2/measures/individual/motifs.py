"""
Mobility motif classification.

Delegates the compute-heavy per-user-per-day graph construction and
canonicalization to the Rust kernel ``_core.compute_daily_motifs``, which
parallelises across users with Rayon.  The Python wrapper handles only column
extraction and result assembly.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from skmob2 import _core

from .._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    DURATION_CANDIDATES,
)


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
    # End-timestamp candidates — motifs-specific, not in _common.py
    _ETS_CANDIDATES = ["end_timestamp", "end_time"]

    nw_df = nw.from_native(df, eager_only=True)
    backend = nw_df.implementation
    cols = nw_df.columns

    # Column auto-detection
    if user_id_col is None:
        user_id_col = _pick_existing_column(cols, USER_ID_CANDIDATES) or "agent_id"
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

    # Build derived columns needed by the Rust kernel:
    #   unique_id   = location_id + "_" + purpose
    #   start_hour  = hour of start_timestamp
    #   end_hour    = hour of end_timestamp
    #   date_id     = days since Unix epoch (Int32) — computed via truncate-to-day
    #                 then divide epoch-microseconds by 86400*1e6
    work_df = work_df.with_columns(
        nw.col("start_timestamp").cast(nw.Datetime("us")).alias("start_timestamp"),
        nw.col("end_timestamp").cast(nw.Datetime("us")).alias("end_timestamp"),
        (nw.col(location_id_col).cast(nw.String) + nw.lit("_") + nw.col("purpose").cast(nw.String)).alias("unique_id"),
    ).with_columns(
        nw.col("start_timestamp").dt.hour().cast(nw.UInt32).alias("start_hour"),
        nw.col("end_timestamp").dt.hour().cast(nw.UInt32).alias("end_hour"),
        (nw.col("start_timestamp").dt.truncate("1d").cast(nw.Int64) // (86400 * 1_000_000))
        .cast(nw.Int32)
        .alias("date_id"),
    )

    # Sort by [user_id, start_timestamp] so visits within each user are
    # in chronological order and users are contiguous in the flat arrays
    work_df = work_df.sort([user_id_col, "start_timestamp"])

    # Extract flat Python lists for the Rust kernel
    user_ids_list: list[str] = work_df[user_id_col].cast(nw.String).to_list()
    unique_ids_list: list[str] = work_df["unique_id"].to_list()
    purposes_list: list[str] = work_df["purpose"].cast(nw.String).to_list()
    start_hours_list: list[int] = work_df["start_hour"].to_list()
    end_hours_list: list[int] = work_df["end_hour"].to_list()
    date_ids_list: list[int] = work_df["date_id"].to_list()

    if "duration_minutes" in work_df.columns:
        durations_list: list[float | None] = work_df["duration_minutes"].to_list()
    else:
        durations_list = [None] * len(user_ids_list)

    # Compute user_ranges: contiguous index ranges for each unique user
    user_ranges: list[tuple[int, int]] = []
    user_id_labels: list[str] = []
    if user_ids_list:
        start = 0
        for i in range(1, len(user_ids_list) + 1):
            if i == len(user_ids_list) or user_ids_list[i] != user_ids_list[start]:
                user_ranges.append((start, i))
                user_id_labels.append(user_ids_list[start])
                start = i

    # Call the Rust kernel
    user_ids_out, date_ids_out, motif_ids_out = _core.compute_daily_motifs(
        unique_ids_list,
        purposes_list,
        start_hours_list,
        end_hours_list,
        date_ids_list,
        durations_list,
        user_ranges,
        user_id_labels,
    )

    if not motif_ids_out:
        empty_daily = nw.from_dict(
            {
                user_id_col: [],
                "date": [],
                "motif_id": [],
                "num_nodes": [],
                "num_edges": [],
            },
            backend=backend,
        )
        empty_dist = nw.from_dict(
            {"motif_id": [], "count": [], "percentage": []},
            backend=backend,
        )
        return empty_daily.to_native(), empty_dist.to_native()

    # Parse num_nodes and num_edges from motif_id strings in Python
    # motif_id format: "m{n_nodes}:{bits}" or "-1"
    num_nodes_list: list[int] = []
    num_edges_list: list[int] = []

    for mid in motif_ids_out:
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

    # Convert date_ids_out (days since epoch) back to Datetime(us)
    # date_id * 86400 * 1_000_000 microseconds → Datetime('us')
    daily_nw = (
        nw.from_dict(
            {
                user_id_col: user_ids_out,
                "date_id_raw": date_ids_out,
                "motif_id": motif_ids_out,
                "num_nodes": num_nodes_list,
                "num_edges": num_edges_list,
            },
            backend=backend,
        )
        .with_columns(
            (nw.col("date_id_raw").cast(nw.Int64) * (86400 * 1_000_000)).cast(nw.Datetime("us")).alias("date")
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
