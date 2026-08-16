"""Amichi et al. (2020) Routiner/Regular/Scouter profile classification.

Distinct from this package's :func:`~fastmob.measures.individual.mobility_profiling.exploration_profiling`:
that function clusters on a single feature (``degree_of_return``) via a
dependency-free Rust KMeans/GMM kernel. This module clusters on **two**
standardized features (``[intermittency, degree_of_return]``) via a native
Rust K-Means kernel, alongside additional per-user descriptive metrics (regularity,
diversity, entropy, stationarity) useful for reporting. The two clusterings
are not expected to produce identical cluster assignments.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.utils._common import (
    ACTIVITY_CANDIDATES,
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _pick_existing_column,
)

from .diversity import diversity
from .entropy import trajectory_entropy
from .mobility_profiling import _END_TIMESTAMP_CANDIDATES, intermittance_and_degree_of_return
from .regularity import regularity

#: Metrics shown in per-profile summaries/box plots.
PROFILE_METRICS = ("regularity", "diversity", "stationarity", "entropy")

_CLUSTER_FEATURES = ["intermittency", "degree_of_return"]
_PROFILE_NAMES = ("routiners", "regulars", "scouters")


def _stationarity(
    visits: nw.DataFrame,
    *,
    user_id_col: str,
    start_col: str,
    end_col: str,
) -> nw.DataFrame:
    """Fraction of observed time each user spends stationary.

    ``stationarity = total dwell time / (last stay end - first stay start)``,
    per user, restricted to stays with positive duration.
    """
    df = visits.with_columns(
        ((nw.col(end_col) - nw.col(start_col)).dt.total_seconds() / 60.0).alias("__duration_minutes__")
    )
    df = df.filter(nw.col("__duration_minutes__") > 0)

    # Two simple aggregations (max, min, sum) rather than one aggregation
    # expression that mixes both columns -- narwhals warns that a "complex"
    # group-by expression like `(col.max() - col.min())` inside `.agg(...)`
    # can't be pushed down efficiently on the pandas backend.
    per_user = df.group_by(user_id_col).agg(
        nw.col(end_col).max().alias("__span_end__"),
        nw.col(start_col).min().alias("__span_start__"),
        nw.col("__duration_minutes__").sum().alias("__dwell__"),
    )
    per_user = per_user.with_columns(
        ((nw.col("__span_end__") - nw.col("__span_start__")).dt.total_seconds() / 60.0).alias("__span__")
    )
    return per_user.select(
        user_id_col,
        nw.when(nw.col("__span__") == 0.0)
        .then(None)
        .otherwise(nw.col("__dwell__") / nw.col("__span__"))
        .alias("stationarity"),
    )


def _cluster_and_label(
    profiles: nw.DataFrame,
    *,
    user_id_col: str,
    n_clusters: int = 3,
    random_state: int = 0,
) -> nw.DataFrame:
    """Cluster users on standardized ``[intermittency, degree_of_return]``
    with KMeans, then label clusters by descending mean degree-of-return
    (highest -> "routiners", lowest -> "scouters"), matching
    :func:`~fastmob.measures.individual.mobility_profiling.exploration_profiling`'s
    naming convention.
    """
    import pyarrow as pa

    from fastmob._core import cluster_standardized_kmeans_arrow
    from fastmob.utils._common import _as_arrow

    first = pa.array(_as_arrow(profiles.get_column(_CLUSTER_FEATURES[0])), type=pa.float64())
    second = pa.array(_as_arrow(profiles.get_column(_CLUSTER_FEATURES[1])), type=pa.float64())
    labels = cluster_standardized_kmeans_arrow(first, second, n_clusters=n_clusters, seed=random_state)
    labels = labels.to_pylist()

    profiles = profiles.with_columns(nw.new_series("__cluster__", labels, backend=profiles.implementation))

    cluster_ids = sorted(set(labels))
    means = {
        cluster_id: float(profiles.filter(nw.col("__cluster__") == cluster_id).get_column("degree_of_return").mean())
        for cluster_id in cluster_ids
    }
    ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
    names = (
        list(_PROFILE_NAMES[:n_clusters])
        if n_clusters <= len(_PROFILE_NAMES)
        else [f"cluster_{i}" for i in range(n_clusters)]
    )
    mapping = {cluster_id: names[rank] for rank, (cluster_id, _mean) in enumerate(ranked)}

    profile_values = [mapping[label] for label in labels]
    profiles = profiles.with_columns(
        nw.new_series("profile", profile_values, backend=profiles.implementation).alias("profile")
    )
    return profiles.drop("__cluster__")


def compute_profiles(
    visits: Any,
    *,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    purpose_col: str | None = None,
    n_clusters: int = 3,
    random_state: int = 0,
) -> Any:
    """Compute per-user mobility-profile metrics and assign profile labels.

    For each user, computes intermittency, degree of return, regularity,
    diversity, (normalized) entropy, and stationarity from a stay-level visit
    table, then clusters users into Routiner/Regular/Scouter profiles
    (Amichi et al. 2020) on standardized ``[intermittency, degree_of_return]``.

    Parameters
    ----------
    visits:
        Stay-level table; any Narwhals-compatible eager backend. Expected
        columns (auto-detected when not given explicitly): a user ID, a
        location ID, a stay start timestamp, a stay end timestamp, and
        optionally a "purpose"/activity type column.
    user_id_col, location_id_col, start_col, end_col, purpose_col:
        Explicit column name overrides; auto-detected when None.
        ``purpose_col`` may be absent entirely (no activity-type column).
    n_clusters:
        Number of clusters to form; must be <= 3 to get named profiles
        ("routiners"/"regulars"/"scouters"), otherwise clusters are named
        ``"cluster_0"``, ``"cluster_1"``, ...
    random_state:
        Seed for KMeans, for reproducible cluster assignments.

    Returns
    -------
    DataFrame
        One row per clustered user with columns ``[user_id_col,
        "intermittency", "degree_of_return", "regularity", "diversity",
        "entropy", "stationarity", "profile"]``, in the same backend as input.

    Raises
    ------
    ValueError
        If fewer than ``n_clusters`` users have finite profiling metrics.
    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.measures.individual import compute_profiles
    >>> rows = []
    >>> patterns = {
    ...     "u1": ["a", "b", "c", "d", "a", "b", "c", "d"],
    ...     "u2": ["a", "b", "a", "c", "a", "b", "a", "d"],
    ...     "u3": ["a", "a", "b", "a", "a", "a", "b", "a"],
    ... }
    >>> for uid, locations in patterns.items():
    ...     for i, location in enumerate(locations):
    ...         rows.append(
    ...             {
    ...                 "uid": uid,
    ...                 "start_timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i),
    ...                 "end_timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i, minutes=30),
    ...                 "location_id": location,
    ...             }
    ...         )
    >>> visits = pd.DataFrame(rows)
    >>> result = compute_profiles(visits, n_clusters=3, random_state=7)
    >>> sorted(result["profile"].unique())
    ['regulars', 'routiners', 'scouters']
    """
    nw_visits = nw.from_native(visits, eager_only=True)
    if user_id_col is None:
        user_id_col = _pick_existing_column(nw_visits.columns, USER_ID_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(nw_visits.columns, LOCATION_CANDIDATES)
    if start_col is None:
        start_col = _pick_existing_column(nw_visits.columns, TIMESTAMP_CANDIDATES)
    if end_col is None:
        end_col = _pick_existing_column(nw_visits.columns, _END_TIMESTAMP_CANDIDATES)
    if purpose_col is None:
        purpose_col = _pick_existing_column(nw_visits.columns, ACTIVITY_CANDIDATES)

    missing = [
        name
        for name, col in [
            ("user_id", user_id_col),
            ("location_id", location_id_col),
            ("start", start_col),
            ("end", end_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not detect required column(s): {missing}. Available columns: {nw_visits.columns}. "
            "Pass the column name(s) explicitly."
        )

    if purpose_col is not None and purpose_col in nw_visits.columns:
        location_token_col = "__location_token__"
        df = nw_visits.with_columns(
            (nw.col(location_id_col).cast(nw.String) + "_" + nw.col(purpose_col).cast(nw.String)).alias(
                location_token_col
            )
        )
    else:
        location_token_col = location_id_col
        df = nw_visits

    df = df.sort([user_id_col, start_col])

    profiles = intermittance_and_degree_of_return(
        df.to_native(), user_id_col=user_id_col, location_id_col=location_token_col, impute_gaps=True
    )
    profiles = nw.from_native(profiles, eager_only=True).select([user_id_col, "intermittency", "degree_of_return"])

    metric_frames = [
        regularity(
            df.to_native(), user_id_col=user_id_col, location_id_col=location_id_col, location_type_col=purpose_col
        ),
        diversity(
            df.to_native(), user_id_col=user_id_col, location_id_col=location_id_col
        ),
        trajectory_entropy(
            df.to_native(),
            user_id_col=user_id_col,
            location_id_col=location_id_col,
            normalized=True,
        ),
        _stationarity(df, user_id_col=user_id_col, start_col=start_col, end_col=end_col).to_native(),
    ]
    for metric_df in metric_frames:
        profiles = profiles.join(nw.from_native(metric_df, eager_only=True), on=user_id_col, how="left")

    # KMeans needs finite clustering features; sparse single-visit users may lack them.
    profiles = profiles.drop_nulls(subset=["intermittency", "degree_of_return"])
    if len(profiles) < n_clusters:
        raise ValueError(f"need at least {n_clusters} users with finite profiling metrics, got {len(profiles)}")

    result = _cluster_and_label(profiles, user_id_col=user_id_col, n_clusters=n_clusters, random_state=random_state)
    return result.to_native()


compute_profiles.__module__ = "fastmob.measures.individual"
