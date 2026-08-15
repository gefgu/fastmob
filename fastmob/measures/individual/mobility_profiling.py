from __future__ import annotations

import math
from typing import Any, Literal, get_args

import narwhals as nw
import numpy as np

from fastmob.utils._common import (
    DATETIME_CANDIDATES,
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _as_arrow,
    _build_indexed_user_ranges,
    _empty_like,
    _extract_timestamp_arrow,
    _extract_timestamps,
    _pick_existing_column,
    _take_uid_values,
    _timestamps_ms_to_datetime_ns,
    _to_native,
)

COLD_START_STRATEGIES = Literal["frequency", "baseline", "max_frequency", "suffix", "none"]
CLUSTERING_METHODS = Literal["kmeans", "gmm"]
_END_TIMESTAMP_CANDIDATES: list[str] = ["end_timestamp", "end_time"]
_TIMESTAMP_CANDIDATES: list[str] = TIMESTAMP_CANDIDATES + [
    col for col in DATETIME_CANDIDATES if col not in TIMESTAMP_CANDIDATES
]
_TRAJECTORY_TIMESTAMP_COL = "__fastmob_trajectory_timestamp__"
#: How many original 5-minute slices a row stands for. Always 1 except for
#: `_expand_to_5min_trajectory`'s `impute_gaps=True` output, where a row can
#: represent a whole run-length-compressed stretch of identical consecutive
#: slices (see that function's docstring). Every place that used to
#: `.count()` rows to get a slice count now `.sum()`s this column instead,
#: so compression changes memory/row-count but never changes a single
#: reported number.
_RUN_LENGTH_COL = "__fastmob_run_length__"


def _expand_to_5min_trajectory(
    nw_df: Any,
    user_id_col: str,
    location_id_col: str,
    start_col: str,
    end_col: str,
    *,
    impute_gaps: bool = False,
) -> Any:
    """Expand each row's ``[start, end]`` interval into 5-minute-aligned slices.

    Routes through Rust kernels in ``fastmob._core``, parallelized per user
    via rayon (``fastmob-core/src/preprocessing/expand_trajectory.rs``):
    send start/end timestamps and the per-user grouping to Rust, get back
    per-slice results, and reconstruct the location column in Python via
    ``pc.take``. Replaces what used to be a pure-Python loop (``.to_list()``
    then a nested ``for``/``while``) that didn't scale -- at 4M input rows it
    produced ~24M expanded rows and made
    ``intermittance_and_degree_of_return``/``exploration_profiling``
    pandas-backend calls run roughly 9x slower than polars at that size, and
    left ``compute_profiles`` (which always calls with ``impute_gaps=True``)
    unusable at that scale.

    ``impute_gaps=False`` (the common case) calls
    ``expand_5min_trajectory_batch_indexed``, which only touches timestamps
    and hands back ``source_row_idx`` for the caller to ``pc.take`` any
    per-row column from the *original* row -- one output row per slice,
    ``_RUN_LENGTH_COL`` is always 1. ``impute_gaps=True`` calls
    ``expand_5min_trajectory_with_imputation_batch_indexed``, which also
    fills gaps between a user's first and last observed slice using
    hour-of-day "anchor" locations; since anchor selection needs location
    values, this variant takes dense-factorized location codes. Its output is
    **run-length-compressed**: one row per maximal run of consecutive
    same-location slices (observed and/or imputed) rather than one row per
    slice, with ``_RUN_LENGTH_COL`` giving each row's real slice count. A
    years-long sparse check-in history can otherwise materialize hundreds of
    thousands of near-identical nightly/workday rows per user; compressed,
    it's one row per run. Every downstream consumer of this output
    (``intermittance_and_degree_of_return``'s block aggregation,
    ``_apply_cold_start_strategy``'s frequency-based branches) sums
    ``_RUN_LENGTH_COL`` instead of counting rows, so compression changes row
    count and memory, never a single reported number -- except
    ``cold_start_strategy="baseline"``, which needs real per-day timestamps
    and can undercount distinct active days for a run that spans multiple
    calendar days; not exercised by any current caller (``compute_profiles``
    always uses the default ``"frequency"``), flagged here rather than
    silently risked.
    """
    import pyarrow.compute as pc

    start_ms = _extract_timestamps(nw_df, start_col)
    end_ms = _extract_timestamps(nw_df, end_col)
    start_timestamps = _extract_timestamp_arrow(nw_df, start_col)
    uid_values, sorted_indices, ends = _build_indexed_user_ranges(
        nw_df, user_id_col, timestamps=start_timestamps
    )

    if impute_gaps:
        from fastmob._core import expand_5min_trajectory_with_imputation_batch_indexed
        from fastmob.utils._common import _factorize_arrow_values

        location_array = _as_arrow(nw_df.get_column(location_id_col))
        location_codes, representative_indices = _factorize_arrow_values(location_array, sort=False)
        # `representative_indices` are row indices into `location_array` (one
        # per unique code), not resolved values -- see `_build_indexed_user_ranges`
        # for the same `pc.take(original_array, representatives)` pattern.
        location_representatives = pc.take(location_array, representative_indices)
        user_range_idx, timestamps_ms, location_codes_out, run_lengths = (
            expand_5min_trajectory_with_imputation_batch_indexed(
                _as_arrow(start_ms), _as_arrow(end_ms), location_codes, sorted_indices, ends
            )
        )
        if len(user_range_idx) == 0:
            empty = _empty_like(nw_df, [user_id_col, _TRAJECTORY_TIMESTAMP_COL, location_id_col, _RUN_LENGTH_COL])
            return nw.from_native(empty, eager_only=True)
        location_values = pc.take(location_representatives, _as_arrow(location_codes_out))
        run_length_values = _as_arrow(run_lengths)
    else:
        from fastmob._core import expand_5min_trajectory_batch_indexed

        user_range_idx, timestamps_ms, source_row_idx = expand_5min_trajectory_batch_indexed(
            _as_arrow(start_ms), _as_arrow(end_ms), sorted_indices, ends
        )
        if len(user_range_idx) == 0:
            empty = _empty_like(nw_df, [user_id_col, _TRAJECTORY_TIMESTAMP_COL, location_id_col, _RUN_LENGTH_COL])
            return nw.from_native(empty, eager_only=True)
        location_values = pc.take(_as_arrow(nw_df.get_column(location_id_col)), _as_arrow(source_row_idx))
        run_length_values = np.ones(len(user_range_idx), dtype=np.uint32)

    out_dict = {
        user_id_col: _take_uid_values(uid_values, user_range_idx),
        _TRAJECTORY_TIMESTAMP_COL: _timestamps_ms_to_datetime_ns(timestamps_ms),
        location_id_col: location_values,
        _RUN_LENGTH_COL: run_length_values,
    }
    return nw.from_native(_to_native(out_dict, nw_df), eager_only=True)


def _apply_cold_start_strategy(
    nw_df: Any,
    user_id_col: str,
    cold_start_strategy: COLD_START_STRATEGIES,
    known_suffixes: tuple[str, ...] = ("_HOME", "_WORK"),
    frequency_level: float = 0.8,
    timestamp_col: str | None = None,
) -> tuple[Any, Any]:
    """Apply cold-start logic and return updated dataframe + known-location expression."""

    if cold_start_strategy == "frequency":
        # Paper's Algorithm 2: Visitation-frequency-based identification.
        # Sums `_RUN_LENGTH_COL` (real slice count per row) rather than
        # counting rows, so run-length-compressed input (impute_gaps=True)
        # yields the same visit frequencies as uncompressed input would.
        loc_counts = nw_df.group_by([user_id_col, "location_key"]).agg(
            nw.col(_RUN_LENGTH_COL).sum().alias("_visit_count")
        )

        # Find the average visitation frequency per user
        mean_counts = loc_counts.group_by(user_id_col).agg(nw.col("_visit_count").mean().alias("_mean_count"))

        # Filter for locations >= mean * level (0.8)
        loc_counts = loc_counts.join(mean_counts, on=user_id_col)
        cold_start_locs = loc_counts.filter(
            nw.col("_visit_count") >= (nw.col("_mean_count") * frequency_level)
        ).with_columns(_is_cold_start=nw.lit(True))

        nw_df = nw_df.join(
            cold_start_locs.select(user_id_col, "location_key", "_is_cold_start"),
            on=[user_id_col, "location_key"],
            how="left",
        )
        is_known_expr = nw.col("_is_cold_start").fill_null(False)

    elif cold_start_strategy == "baseline":
        # Paper's Algorithm 1: Papandrea et al. Relevance score
        if timestamp_col is None:
            raise ValueError("The 'baseline' strategy requires a 'timestamp_col' to calculate daily relevance.")

        # Truncate datetime to day boundary so n_unique() counts distinct active days.
        # dt.truncate("1d") is cross-backend (pandas + polars) unlike dt.date().
        nw_df = nw_df.with_columns(_date=nw.col(timestamp_col).dt.truncate("1d"))

        # d_visit(id_i, u): Number of days user visited the specific location
        d_visit = nw_df.group_by([user_id_col, "location_key"]).agg(nw.col("_date").n_unique().alias("d_visit"))

        # d_total(u): Total number of days the user has been active
        d_total = nw_df.group_by(user_id_col).agg(nw.col("_date").n_unique().alias("d_total"))

        # Calculate Relevance R_u(id_i) = d_visit / d_total
        relevance_df = d_visit.join(d_total, on=user_id_col).with_columns(R_u=nw.col("d_visit") / nw.col("d_total"))

        # Papandrea et al. Algorithm 1, line 6: k-means with 3 components classifies
        # each user's locations into MVP (Mostly Visited Places), OVP (Occasionally
        # Visited Places), and EVP (Exceptionally Visited Places) by R_u score.
        # Cold-start "known" places = MVP (highest R_u cluster).
        from fastmob._core import cluster_kmeans as _cluster_kmeans

        known_rows = []
        relevance_df = relevance_df.sort(user_id_col)
        uid_values = relevance_df.get_column(user_id_col).to_list()
        loc_values = relevance_df.get_column("location_key").to_list()
        r_all = relevance_df.get_column("R_u").to_list()

        start = 0
        while start < len(uid_values):
            end = start + 1
            while end < len(uid_values) and uid_values[end] == uid_values[start]:
                end += 1
            uid = uid_values[start]
            r_values = r_all[start:end]
            loc_keys = loc_values[start:end]
            n = len(r_values)
            k = min(3, n)  # paper uses 3 (MVP/OVP/EVP); fall back gracefully
            labels = _cluster_kmeans(r_values, n_clusters=k)
            cluster_means = {c: np.mean([v for v, label in zip(r_values, labels) if label == c]) for c in set(labels)}
            # MVP = cluster with highest mean R_u
            mvp_cluster = max(cluster_means, key=cluster_means.get)
            for loc_key, label in zip(loc_keys, labels):
                if label == mvp_cluster:
                    known_rows.append({user_id_col: uid, "location_key": loc_key})
            start = end

        if known_rows:
            known_nw = nw.from_dict(
                {
                    user_id_col: [row[user_id_col] for row in known_rows],
                    "location_key": [row["location_key"] for row in known_rows],
                    "_is_cold_start": [True] * len(known_rows),
                },
                backend=nw_df.implementation,
            )
            nw_df = nw_df.join(
                known_nw.select(user_id_col, "location_key", "_is_cold_start"),
                on=[user_id_col, "location_key"],
                how="left",
            )
        else:
            nw_df = nw_df.with_columns(_is_cold_start=nw.lit(False))

        nw_df = nw_df.drop("_date")
        is_known_expr = nw.col("_is_cold_start").fill_null(False)

    elif cold_start_strategy == "max_frequency":
        # Your original 90% of max implementation. Sums `_RUN_LENGTH_COL`
        # for the same reason the "frequency" branch above does.
        loc_counts = nw_df.group_by([user_id_col, "location_key"]).agg(
            nw.col(_RUN_LENGTH_COL).sum().alias("_visit_count")
        )

        max_counts = loc_counts.group_by(user_id_col).agg(nw.col("_visit_count").max().alias("_max_count"))

        loc_counts = loc_counts.join(max_counts, on=user_id_col)
        cold_start_locs = loc_counts.filter(nw.col("_visit_count") >= (nw.col("_max_count") * 0.9)).with_columns(
            _is_cold_start=nw.lit(True)
        )

        nw_df = nw_df.join(
            cold_start_locs.select(user_id_col, "location_key", "_is_cold_start"),
            on=[user_id_col, "location_key"],
            how="left",
        )
        is_known_expr = nw.col("_is_cold_start").fill_null(False)

    elif cold_start_strategy == "suffix":
        is_known_expr = nw.lit(False)
        for suffix in known_suffixes:
            is_known_expr = is_known_expr | nw.col("location_key").str.ends_with(suffix)

    else:
        is_known_expr = nw.lit(False)

    return nw_df, is_known_expr


def intermittance_and_degree_of_return(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    datetime_col: str | None = None,
    cold_start_strategy: COLD_START_STRATEGIES = "frequency",
    known_suffixes: tuple[str, ...] = ("_HOME", "_WORK"),
    use_trajectory: bool = True,
    impute_gaps: bool = False,
) -> Any:
    """Compute intermittancy and degree of return per user using vectorized operations.

    For each user, partitions the visit sequence into alternating blocks of
    *explorations* (new places) and *returns* (revisits or home/work visits).
    Then computes summary statistics over those blocks. By default, stay
    intervals are reconstructed into 5-minute trajectory slices when start and
    end timestamp columns are available.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if None.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if None.
    datetime_col : str or None, optional
        Column name for visit timestamps. Auto-detected if None. Required when
        ``cold_start_strategy="baseline"`` (used to count distinct active days).
        Also used as the stay start timestamp when ``use_trajectory=True``.
    cold_start_strategy : str, optional
        How to initialize "known" places.

        * ``"frequency"`` — locations visited at or above mean frequency (×0.8).
        * ``"baseline"`` — Papandrea et al. Algorithm 1: k-means (k=3) on the
          per-location relevance score ``R_u = d_visit / d_total``; MVP cluster
          (most frequently visited) is pre-labelled as known. Requires a datetime
          column.
        * ``"max_frequency"`` — top 10 % by visit count.
        * ``"suffix"`` — locations whose ID ends with any of ``known_suffixes``.
        * ``"none"`` — no cold-start; every first visit is an exploration.
    known_suffixes : tuple of str, optional
        Used only when ``cold_start_strategy="suffix"``.
    use_trajectory : bool, optional
        If True, and both start and end timestamp columns are available, expand
        each stay into observed 5-minute slices from ``ceil(start, "5min")`` to
        ``floor(end, "5min")``. Duplicate
        ``(user, timestamp)`` slices keep the first input occurrence. If no end
        timestamp column is available, the function falls back to treating each
        input row as one sequence event. If False, each input row is always one
        sequence event.
    impute_gaps : bool, optional
        If True, and trajectory reconstruction is possible, fill missing
        5-minute slices between each user's first and last observed slice using
        per-user anchors inferred from observed locations: hours 2-5 use the
        most frequent nighttime location, hour 10 uses the most frequent
        location at 10, and hours 14-16 use the most frequent afternoon
        location. Missing slices outside those windows, or inside a window with
        no inferred anchor, remain absent. Ignored when ``use_trajectory=False``
        or when no end timestamp column is available.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns
        ``[user_id_col, "intermittency", "degree_of_return", "mean_return",
        "mean_exploration"]``. The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import intermittance_and_degree_of_return
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
    ...         "start_timestamp": pd.to_datetime(
    ...             [
    ...                 "2020-01-01 08:00",
    ...                 "2020-01-01 09:00",
    ...                 "2020-01-01 18:00",
    ...                 "2020-01-02 08:00",
    ...                 "2020-01-01 07:30",
    ...                 "2020-01-01 12:00",
    ...                 "2020-01-01 19:00",
    ...             ]
    ...         ),
    ...         "location_id": ["home", "work", "home", "gym", "home", "shop", "home"],
    ...     }
    ... )
    >>> result = intermittance_and_degree_of_return(visits, cold_start_strategy="none")
    >>> print(result.round(3).to_string(index=False))
    user_id  intermittency  degree_of_return  mean_return  mean_exploration
         u1            2.5             0.588          1.0               1.5
         u2            3.0             0.464          1.0               2.0
    """
    nw_df = nw.from_native(visits, eager_only=True)

    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_df.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_df.columns, LOCATION_CANDIDATES)
    if datetime_col is None:
        datetime_col = _pick_existing_column(nw_df.columns, _TIMESTAMP_CANDIDATES)

    timestamp_col = datetime_col
    if use_trajectory:
        end_timestamp_col = _pick_existing_column(nw_df.columns, _END_TIMESTAMP_CANDIDATES)
        if datetime_col is not None and end_timestamp_col is not None:
            nw_df = _expand_to_5min_trajectory(
                nw_df=nw_df,
                user_id_col=user_id_col,
                location_id_col=location_id_col,
                start_col=datetime_col,
                end_col=end_timestamp_col,
                impute_gaps=impute_gaps,
            )
            timestamp_col = _TRAJECTORY_TIMESTAMP_COL

    # Only `_expand_to_5min_trajectory`'s impute_gaps=True output carries a
    # real (possibly >1) `_RUN_LENGTH_COL`; every other path (no trajectory
    # expansion, or impute_gaps=False) is one row per slice, so default it
    # to a literal 1 here rather than duplicating that default at every
    # `.sum(_RUN_LENGTH_COL)` call site below.
    if _RUN_LENGTH_COL not in nw_df.columns:
        nw_df = nw_df.with_columns(nw.lit(1).alias(_RUN_LENGTH_COL))

    # 1. Build the unique location key
    location_expr = nw.col(location_id_col).cast(nw.String).fill_null("unknown")
    nw_df = nw_df.with_columns(location_key=location_expr)

    # 2. Assign an absolute sequential row index to preserve the temporal order natively
    nw_df = nw_df.with_columns(_dummy=nw.lit(1)).with_columns(_row_idx=nw.col("_dummy").cum_sum()).drop("_dummy")

    # 3. Find the global first occurrence of each location per user
    first_visits = nw_df.group_by([user_id_col, "location_key"]).agg(nw.col("_row_idx").min().alias("_first_idx"))
    nw_df = nw_df.join(first_visits, on=[user_id_col, "location_key"])

    # 4. State tracking: apply cold-start strategy
    nw_df, is_known_expr = _apply_cold_start_strategy(
        nw_df=nw_df,
        user_id_col=user_id_col,
        cold_start_strategy=cold_start_strategy,
        known_suffixes=known_suffixes,
        timestamp_col=timestamp_col,
    )

    # 5. A place is known if its current index > its first seen index, OR if it triggers the cold start
    nw_df = nw_df.with_columns(is_known=(nw.col("_row_idx") > nw.col("_first_idx")) | is_known_expr)

    # 6. Sort chronologically per user to isolate sequential blocks
    nw_df = nw_df.sort([user_id_col, "_row_idx"])

    # 7. Detect state alternations (Exploration <-> Return) using rolling shifts
    nw_df = nw_df.with_columns(
        is_known_shifted=nw.col("is_known").shift(1), user_id_shifted=nw.col(user_id_col).shift(1)
    )

    nw_df = nw_df.with_columns(
        block_change=(
            (nw.col("is_known") != nw.col("is_known_shifted")) | (nw.col(user_id_col) != nw.col("user_id_shifted"))
        )
        .fill_null(True)
        .cast(nw.Int32)
    )

    # Generate isolated block IDs
    nw_df = nw_df.with_columns(block_id=nw.col("block_change").cum_sum())

    # 8. Aggregate block volumes (counting pure visits as per the paper, discarding duration).
    # Sums `_RUN_LENGTH_COL` rather than counting rows so run-length-compressed
    # blocks (impute_gaps=True) report the same lengths uncompressed rows would.
    blocks = nw_df.group_by([user_id_col, "block_id", "is_known"]).agg(
        nw.col(_RUN_LENGTH_COL).sum().alias("block_length")
    )

    # 9. Compute the mean sequential lengths (#U and #R)
    explorations = (
        blocks.filter(~nw.col("is_known"))
        .group_by(user_id_col)
        .agg(nw.col("block_length").mean().alias("mean_exploration"))
    )
    returns = (
        blocks.filter(nw.col("is_known")).group_by(user_id_col).agg(nw.col("block_length").mean().alias("mean_return"))
    )

    # 10. Merge aggregations per unique user and zero-fill null edge cases
    users = nw_df.select(user_id_col).unique()
    res = users.join(explorations, on=user_id_col, how="left").join(returns, on=user_id_col, how="left")

    res = res.with_columns(nw.col("mean_exploration").fill_null(0.0), nw.col("mean_return").fill_null(0.0))

    # 11. Intermittency calculation directly in Narwhals
    res = res.with_columns(intermittency=(nw.col("mean_exploration") + nw.col("mean_return")).alias("intermittency"))

    # 12. Convert to a pure Python dictionary of lists
    data_dict = res.to_dict(as_series=False)

    # 13. Compute degree_of_return using standard Python math
    data_dict["degree_of_return"] = [
        math.atan2(r, e) for r, e in zip(data_dict["mean_return"], data_dict["mean_exploration"])
    ]

    # 14. Reconstruct the Narwhals DataFrame, preserving the original backend
    final_nw_df = nw.from_dict(
        data_dict,
        backend=nw.get_native_namespace(res),
    )

    # Return final structured DataFrame safely to its native type
    return final_nw_df.select(
        user_id_col, "intermittency", "degree_of_return", "mean_return", "mean_exploration"
    ).to_native()


def exploration_profiling(
    visits: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    datetime_col: str | None = None,
    cold_start_strategy: COLD_START_STRATEGIES = "frequency",
    known_suffixes: tuple[str, ...] = ("_HOME", "_WORK"),
    clustering_method: CLUSTERING_METHODS = "kmeans",
    random_seed: int = 42,
    n_iterations: int = 300,
    impute_gaps: bool = False,
) -> Any:
    """Compute intermittency and degree of return, then cluster users into mobility profiles.

    Calls :func:`intermittance_and_degree_of_return` to get per-user statistics, then
    applies a Rust clustering kernel on the ``degree_of_return`` values to assign each
    user to one of three profiles:

    * **routiners** — high degree of return; mostly revisit familiar places.
    * **regulars** — balanced mix of exploration and return.
    * **scouters** — low degree of return; mostly explore new places.

    Parameters
    ----------
    visits : DataFrame-like
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col : str or None, optional
        Column name for the user ID. Auto-detected if ``None``.
    location_id_col : str or None, optional
        Column name for the location ID. Auto-detected if ``None``.
    datetime_col : str or None, optional
        Column name for visit timestamps. Auto-detected if ``None``. Required
        when ``cold_start_strategy="baseline"``.
    cold_start_strategy : str, optional
        How to initialize known places. Passed directly to
        :func:`intermittance_and_degree_of_return`.
    known_suffixes : tuple of str, optional
        Location ID suffixes treated as known (only used when
        ``cold_start_strategy="suffix"``).
    clustering_method : str, optional
        ``"kmeans"`` for K-Means or ``"gmm"`` for Gaussian Mixture Model.
    random_seed : int, optional
        Seed for the clustering algorithm — ensures reproducible results.
    n_iterations : int, optional
        Maximum iterations for the clustering algorithm.
    impute_gaps : bool, optional
        If True, pass through to :func:`intermittance_and_degree_of_return` to
        fill eligible missing 5-minute trajectory slices before computing
        return/exploration statistics.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[user_id_col, "intermittency",
        "degree_of_return", "mean_return", "mean_exploration", "profile"]``.
        The ``"profile"`` column contains one of ``"routiners"``,
        ``"regulars"``, or ``"scouters"``. The return type matches the input
        backend (pandas, polars, …).

    Raises
    ------
    ValueError
        If fewer than 3 users are present (cannot form 3 clusters), or if
        an unknown ``clustering_method`` is given.
    ImportError
        If the compiled ``fastmob._core`` extension is not available (run
        ``maturin develop`` first).

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import exploration_profiling
    >>> rows = []
    >>> patterns = {
    ...     "u1": ["a", "b", "c", "d", "a"],
    ...     "u2": ["a", "b", "a", "c", "a", "d"],
    ...     "u3": ["a", "a", "b", "a", "c", "a"],
    ...     "u4": ["a", "b", "c", "a", "b", "c"],
    ...     "u5": ["a", "b", "a", "b", "a", "b"],
    ... }
    >>> for uid, locations in patterns.items():
    ...     for i, location in enumerate(locations):
    ...         rows.append(
    ...             {
    ...                 "user_id": uid,
    ...                 "start_timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i),
    ...                 "location_id": location,
    ...             }
    ...         )
    >>> visits = pd.DataFrame(rows)
    >>> result = exploration_profiling(visits, cold_start_strategy="none", random_seed=7)
    >>> print(result.round(3).to_string(index=False))
    user_id  intermittency  degree_of_return  mean_return  mean_exploration   profile
         u1          5.000             0.245          1.0             4.000  scouters
         u2          2.333             0.644          1.0             1.333  regulars
         u3          2.000             0.785          1.0             1.000  regulars
         u4          6.000             0.785          3.0             3.000  regulars
         u5          6.000             1.107          4.0             2.000 routiners
    """
    from fastmob._core import cluster_gmm, cluster_kmeans

    valid_methods = get_args(CLUSTERING_METHODS)
    if clustering_method not in valid_methods:
        raise ValueError(f"Unknown clustering_method {clustering_method!r}. Choose one of {valid_methods}.")

    nw_visits = nw.from_native(visits, eager_only=True)
    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_visits.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_visits.columns, LOCATION_CANDIDATES)

    stats = intermittance_and_degree_of_return(
        visits,
        user_id_col=user_id_col,
        location_id_col=location_id_col,
        datetime_col=datetime_col,
        cold_start_strategy=cold_start_strategy,
        known_suffixes=known_suffixes,
        impute_gaps=impute_gaps,
    )

    n_users = len(stats) if hasattr(stats, "__len__") else stats.shape[0]
    if n_users < 3:
        raise ValueError(f"exploration_profiling requires at least 3 users to form 3 clusters, got {n_users}.")

    stats_nw = nw.from_native(stats, eager_only=True)
    dor_values: list[float] = stats_nw["degree_of_return"].to_list()

    if clustering_method == "kmeans":
        labels: list[int] = cluster_kmeans(dor_values, n_clusters=3, max_iter=n_iterations, seed=random_seed)
    else:
        labels = cluster_gmm(dor_values, n_clusters=3, max_iter=n_iterations, seed=random_seed)

    # Map numeric labels to profile names: sort clusters by mean degree_of_return.
    # Lowest mean → "scouters", middle → "regulars", highest → "routiners".
    cluster_ids = sorted(set(labels))
    cluster_means = {c: float(np.mean([v for v, label in zip(dor_values, labels) if label == c])) for c in cluster_ids}
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
    label_to_profile = {
        sorted_clusters[0][0]: "scouters",
        sorted_clusters[1][0]: "regulars",
        sorted_clusters[2][0]: "routiners",
    }
    profiles = [label_to_profile[label] for label in labels]

    ns = nw.get_native_namespace(stats_nw)
    data = stats_nw.to_dict(as_series=False)
    data["profile"] = profiles
    return nw.from_dict(data, backend=ns).to_native()
