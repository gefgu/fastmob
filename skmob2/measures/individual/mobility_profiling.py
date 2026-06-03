from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal, get_args

import math
import numpy as np
import narwhals as nw

from .._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    DATETIME_CANDIDATES,
    TIMESTAMP_CANDIDATES,
)

COLD_START_STRATEGIES = Literal["frequency", "baseline", "max_frequency", "suffix", "none"]
CLUSTERING_METHODS = Literal["kmeans", "gmm"]
_END_TIMESTAMP_CANDIDATES: list[str] = ["end_timestamp", "end_time"]
_FIVE_MINUTES = timedelta(minutes=5)
_TIMESTAMP_CANDIDATES: list[str] = TIMESTAMP_CANDIDATES + [
    col for col in DATETIME_CANDIDATES if col not in TIMESTAMP_CANDIDATES
]
_TRAJECTORY_TIMESTAMP_COL = "__skmob2_trajectory_timestamp__"


def _floor_5min(dt: Any) -> Any:
    if hasattr(dt, "floor"):
        return dt.floor("5min")
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _ceil_5min(dt: Any) -> Any:
    if hasattr(dt, "ceil"):
        return dt.ceil("5min")
    floored = _floor_5min(dt)
    return floored if floored == dt else floored + _FIVE_MINUTES


def _expand_to_5min_trajectory(
    nw_df: Any,
    user_id_col: str,
    location_id_col: str,
    start_col: str,
    end_col: str,
) -> Any:
    user_values = nw_df.get_column(user_id_col).to_list()
    location_values = nw_df.get_column(location_id_col).to_list()
    start_values = nw_df.get_column(start_col).to_list()
    end_values = nw_df.get_column(end_col).to_list()

    rows: list[tuple[Any, Any, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for uid, location, start, end in zip(user_values, location_values, start_values, end_values):
        if start is None or end is None:
            continue

        timestamp = _ceil_5min(start)
        last_timestamp = _floor_5min(end)
        while timestamp <= last_timestamp:
            key = (uid, timestamp)
            if key not in seen:
                rows.append((uid, timestamp, location))
                seen.add(key)
            timestamp = timestamp + _FIVE_MINUTES

    rows.sort(key=lambda row: (str(row[0]), row[1]))
    return nw.from_dict(
        {
            user_id_col: [row[0] for row in rows],
            _TRAJECTORY_TIMESTAMP_COL: [row[1] for row in rows],
            location_id_col: [row[2] for row in rows],
        },
        backend=nw_df.implementation,
    )


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
        # Paper's Algorithm 2: Visitation-frequency-based identification
        loc_counts = nw_df.group_by([user_id_col, "location_key"]).agg(
            nw.col("location_key").count().alias("_visit_count")
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
        from skmob2._core import cluster_kmeans as _cluster_kmeans

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
        # Your original 90% of max implementation
        loc_counts = nw_df.group_by([user_id_col, "location_key"]).agg(
            nw.col("location_key").count().alias("_visit_count")
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
) -> Any:
    """Compute intermittancy and degree of return per user using vectorized operations.

    For each user, partitions the visit sequence into alternating blocks of
    *explorations* (new places) and *returns* (revisits or home/work visits).
    Then computes summary statistics over those blocks. By default, stay
    intervals are reconstructed into 5-minute trajectory slices when start and
    end timestamp columns are available.

    Parameters
    ----------
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if None.
    location_id_col:
        Column name for the location ID. Auto-detected if None.
    datetime_col:
        Column name for visit timestamps. Auto-detected if None. Required when
        ``cold_start_strategy="baseline"`` (used to count distinct active days).
        Also used as the stay start timestamp when ``use_trajectory=True``.
    cold_start_strategy:
        How to initialize "known" places.

        * ``"frequency"`` — locations visited at or above mean frequency (×0.8).
        * ``"baseline"`` — Papandrea et al. Algorithm 1: k-means (k=3) on the
          per-location relevance score ``R_u = d_visit / d_total``; MVP cluster
          (most frequently visited) is pre-labelled as known. Requires a datetime
          column.
        * ``"max_frequency"`` — top 10 % by visit count.
        * ``"suffix"`` — locations whose ID ends with any of ``known_suffixes``.
        * ``"none"`` — no cold-start; every first visit is an exploration.
    known_suffixes:
        Used only when ``cold_start_strategy="suffix"``.
    use_trajectory:
        If True, and both start and end timestamp columns are available, expand
        each stay into observed 5-minute slices from ``ceil(start, "5min")`` to
        ``floor(end, "5min")`` without imputing missing gaps. Duplicate
        ``(user, timestamp)`` slices keep the first input occurrence. If no end
        timestamp column is available, the function falls back to treating each
        input row as one sequence event. If False, each input row is always one
        sequence event.

    Returns
    -------
    DataFrame
        One row per user with columns
        ``[user_id_col, "intermittency", "degree_of_return", "mean_return",
        "mean_exploration"]``. The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.individual import intermittance_and_degree_of_return
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
            )
            timestamp_col = _TRAJECTORY_TIMESTAMP_COL

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

    # 8. Aggregate block volumes (counting pure visits as per the paper, discarding duration)
    blocks = nw_df.group_by([user_id_col, "block_id", "is_known"]).agg(nw.col("block_id").count().alias("block_length"))

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
    visits:
        A DataFrame (any Narwhals-compatible backend) with visit rows.
    user_id_col:
        Column name for the user ID. Auto-detected if ``None``.
    location_id_col:
        Column name for the location ID. Auto-detected if ``None``.
    datetime_col:
        Column name for visit timestamps. Auto-detected if ``None``. Required
        when ``cold_start_strategy="baseline"``.
    cold_start_strategy:
        How to initialize known places. Passed directly to
        :func:`intermittance_and_degree_of_return`.
    known_suffixes:
        Location ID suffixes treated as known (only used when
        ``cold_start_strategy="suffix"``).
    clustering_method:
        ``"kmeans"`` for K-Means or ``"gmm"`` for Gaussian Mixture Model.
    random_seed:
        Seed for the clustering algorithm — ensures reproducible results.
    n_iterations:
        Maximum iterations for the clustering algorithm.

    Returns
    -------
    DataFrame
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
        If the compiled ``skmob2._core`` extension is not available (run
        ``maturin develop`` first).

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.individual import exploration_profiling
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
    from skmob2._core import cluster_kmeans, cluster_gmm

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
