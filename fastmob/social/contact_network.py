"""Co-presence / contact-network construction, validation, and social-tie inference.

Builds a contact network from raw mobility data (who was where, when),
computes population-level graph metrics for validating it against a
degree-preserving random-graph null model, and infers which specific
co-presence pairs represent genuine social ties rather than incidental
contact (a Fournet & Barrat-style random-graph-baseline significance test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

from fastmob.utils._common import (
    DATETIME_CANDIDATES,
    LOCATION_CANDIDATES,
    USER_ID_CANDIDATES,
    _factorize_uids_uint64,
    _pick_existing_column,
)


@dataclass(frozen=True)
class NetworkGraph:
    """Undirected graph as a plain edge list (``u < v``, sorted, deduped),
    not per-node adjacency ``set``s -- for a real-world co-presence graph
    (tens of millions of edges), materializing a Python `set`/`set`-of-`set`s
    costs seconds of object construction and gigabytes of memory on its own,
    on top of the O(sum of degree^2) metric loops that would follow.
    ``clustering_coefficients``/``topological_overlap`` consume these arrays
    directly via the Rust extension; ``.edges`` is a convenience for small
    graphs (tests, synthetic graphs) and should not be used in a hot path
    over a large observed graph.
    """

    node_count: int
    edge_from: np.ndarray  # uint32[E], edge_from < edge_to elementwise
    edge_to: np.ndarray  # uint32[E]

    @property
    def edge_count(self) -> int:
        return int(self.edge_from.shape[0])

    @property
    def edges(self) -> set[tuple[int, int]]:
        return set(zip(self.edge_from.tolist(), self.edge_to.tolist()))

    def degrees(self) -> np.ndarray:
        return (
            np.bincount(
                np.concatenate([self.edge_from, self.edge_to]),
                minlength=self.node_count,
            )
            if self.edge_from.size
            else np.zeros(self.node_count, dtype=np.int64)
        )


def _empty_graph(node_count: int) -> NetworkGraph:
    empty = np.empty(0, dtype=np.uint32)
    return NetworkGraph(node_count=node_count, edge_from=empty, edge_to=empty)


def _normal_edge(a: Any, b: Any, node_count: int) -> tuple[int, int] | None:
    try:
        u, v = int(a), int(b)
    except (TypeError, ValueError):
        return None
    if u == v or u < 0 or v < 0 or u >= node_count or v >= node_count:
        return None
    return (u, v) if u < v else (v, u)


def graph_from_edges(node_count: int, edges: set[tuple[int, int]]) -> NetworkGraph:
    """Build a graph from a small/moderate edge collection (synthetic-scale
    social graphs, test fixtures) -- normalizes/dedupes via a Python ``set``
    since that's cheap at this scale. For real observed co-presence data at
    scale, use :func:`co_presence_graph_from_visits` instead, which builds
    the edge list directly via the Rust extension.
    """
    normalized: set[tuple[int, int]] = set()
    for u, v in edges:
        edge = _normal_edge(u, v, node_count)
        if edge is not None:
            normalized.add(edge)
    if not normalized:
        return _empty_graph(node_count)
    ordered = sorted(normalized)
    edge_from = np.ascontiguousarray([u for u, _ in ordered], dtype=np.uint32)
    edge_to = np.ascontiguousarray([v for _, v in ordered], dtype=np.uint32)
    return NetworkGraph(node_count=node_count, edge_from=edge_from, edge_to=edge_to)


def co_presence_graph_from_visits(
    visits: Any,
    *,
    user_id_col: str | None = None,
    datetime_col: str | None = None,
    location_id_col: str | None = None,
    day_col: str | None = None,
    max_group_size: int = 200,
) -> tuple[NetworkGraph, np.ndarray, int, dict[str, int]]:
    """Build a co-presence graph directly from raw mobility data.

    Two users are "co-present" on a given day if they were observed at the
    same location on that day. Groups ``(day, location)`` presence rows and
    emits one edge per unique co-presence pair, with per-edge persistence
    (fraction of distinct days the pair was seen together on) -- via the
    Rust extension, which scales to real-world observed data (tens of
    millions of raw pair-instances) where a pure-Python
    ``itertools.combinations`` loop would not.

    Parameters
    ----------
    visits:
        Mobility data with at least a user ID, a timestamp, and a location
        ID; any Narwhals-compatible eager backend.
    user_id_col, datetime_col, location_id_col:
        Explicit column name overrides; auto-detected when None.
    day_col:
        Explicit "day" column (any hashable value grouping rows into the
        same observation day); when None, derived from ``datetime_col`` by
        truncating to midnight.
    max_group_size:
        Skip (day, location) groups larger than this rather than emitting
        `O(group_size^2)` edges for an uninformatively large crowd. Must be
        at least 2.

    Returns
    -------
    graph, persistence, time_steps, skip_info:
        ``graph`` is a :class:`NetworkGraph` over ``node_count`` = number of
        distinct users; node indices are dense integers in sort order of the
        user ID column (not the original labels -- track your own mapping
        if you need it back). ``persistence[i]`` is edge ``i``'s fraction of
        ``time_steps`` (distinct days) the pair co-occurred on. ``skip_info``
        is ``{"skipped_groups": int, "skipped_rows": int}``: how many
        (day, location) groups (and total user-presences within them)
        exceeded ``max_group_size`` and were skipped entirely.

    Raises
    ------
    ValueError
        If required columns cannot be found, or ``max_group_size < 2``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.social import co_presence_graph_from_visits
    >>> visits = pd.DataFrame(
    ...     {
    ...         "uid": ["a", "b", "c", "a", "b"],
    ...         "datetime": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2),
    ...         "location_id": ["venue1", "venue1", "venue1", "venue1", "venue1"],
    ...     }
    ... )
    >>> graph, persistence, time_steps, skip_info = co_presence_graph_from_visits(visits)
    >>> graph.edge_count
    3
    >>> time_steps
    2
    """
    if max_group_size < 2:
        raise ValueError("max_group_size must be at least 2")

    df = nw.from_native(visits, eager_only=True)
    if user_id_col is None:
        user_id_col = _pick_existing_column(df.columns, USER_ID_CANDIDATES)
    if datetime_col is None:
        datetime_col = _pick_existing_column(df.columns, DATETIME_CANDIDATES)
    if location_id_col is None:
        location_id_col = _pick_existing_column(df.columns, LOCATION_CANDIDATES)

    missing = [name for name, col in [("user_id", user_id_col), ("location_id", location_id_col)] if col is None]
    if day_col is None and datetime_col is None:
        missing.append("datetime (or day_col)")
    if missing:
        raise ValueError(
            f"Could not detect required column(s): {missing}. Available columns: {df.columns}. "
            "Pass the column name(s) explicitly."
        )

    if day_col is None:
        df = df.with_columns(nw.col(datetime_col).dt.truncate("1d").alias("__day__"))
        day_col = "__day__"

    work = df.select([user_id_col, day_col, location_id_col]).drop_nulls()
    if len(work) == 0:
        return _empty_graph(0), np.asarray([], dtype=float), 0, {"skipped_groups": 0, "skipped_rows": 0}

    # Vectorized per-backend factorization (pandas: pd.factorize, polars:
    # replace_strict, pyarrow: dictionary_encode) -- not a per-row Python
    # dict lookup, which wouldn't scale to real observed data (tens of
    # millions of rows).
    uid_codes, node_count = _factorize_uids_uint64(work, user_id_col, sort=True)
    day_codes, time_steps = _factorize_uids_uint64(work, day_col, sort=True)
    location_codes, _n_locations = _factorize_uids_uint64(work, location_id_col, sort=False)

    coded = work.with_columns(
        uid_codes.cast(nw.Int64).alias("__uid_code__"),
        day_codes.cast(nw.Int64).alias("__day_code__"),
        location_codes.cast(nw.Int64).alias("__location_code__"),
    ).unique(subset=["__uid_code__", "__day_code__", "__location_code__"])

    from fastmob._core import build_co_presence_edges

    edge_from, edge_to, persistence, skipped_groups, skipped_rows = build_co_presence_edges(
        coded.get_column("__day_code__").to_numpy().astype(np.int64),
        coded.get_column("__location_code__").to_numpy().astype(np.int64),
        coded.get_column("__uid_code__").to_numpy().astype(np.int64),
        max_group_size,
        time_steps,
    )
    graph = NetworkGraph(node_count=node_count, edge_from=edge_from, edge_to=edge_to)
    skip_info = {"skipped_groups": int(skipped_groups), "skipped_rows": int(skipped_rows)}
    return graph, persistence, time_steps, skip_info


def co_presence_graph_from_staypoints(
    staypoints: Any,
    *,
    locations: Any | None = None,
    location_id_col: str = "location_id",
    day_col: str | None = None,
    max_group_size: int = 200,
) -> tuple[NetworkGraph, np.ndarray, int, dict[str, int]]:
    """Build a co-presence graph from staypoints assigned to global locations.

    ``locations``, when supplied, must be a global :class:`Locations`
    catalogue. The staypoints must already carry matching exact IDs; use
    :meth:`Staypoints.associate_global_locations` to validate an external
    catalogue first. The legacy dataframe function remains available for
    callers with generic visitation tables.
    """
    from fastmob.core.staypoints_dataframe import Staypoints

    if isinstance(staypoints, Staypoints):
        if locations is not None:
            staypoints = staypoints.associate_global_locations(locations, location_id_col=location_id_col)
        return co_presence_graph_from_visits(
            staypoints.df,
            user_id_col=staypoints.uid_col,
            datetime_col=staypoints.started_at_col,
            location_id_col=location_id_col,
            day_col=day_col,
            max_group_size=max_group_size,
        )
    if locations is not None and locations.scope != "global":
        raise ValueError("co_presence_graph_from_staypoints requires global Locations")
    return co_presence_graph_from_visits(
        staypoints, location_id_col=location_id_col, day_col=day_col, max_group_size=max_group_size
    )


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    """Per-node clustering coefficient (see :func:`_graph_metrics`)."""
    clustering, _overlap = _graph_metrics(graph)
    return clustering


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    """Per-edge topological overlap / Jaccard similarity (see :func:`_graph_metrics`)."""
    _clustering, overlap = _graph_metrics(graph)
    return overlap


def _graph_metrics(graph: NetworkGraph) -> tuple[np.ndarray, np.ndarray]:
    from fastmob._core import graph_metrics as _graph_metrics_core

    return _graph_metrics_core(graph.node_count, graph.edge_from, graph.edge_to)


def degree_preserving_random_graph(
    degrees: np.ndarray,
    *,
    seed: int = 42,
) -> NetworkGraph:
    """Chung-Lu-style degree-preserving random graph: each pair `(i, j)` is
    an edge with probability proportional to `degree[i] * degree[j]`, so the
    expected degree sequence approximately matches `degrees`. Used as a null
    model to check whether an observed graph's metrics (clustering
    coefficient, topological overlap, edge persistence) differ from what
    degree alone would produce.
    """
    deg = np.asarray(degrees, dtype=float)
    n = len(deg)
    total_degree = float(deg.sum())
    if n <= 1 or total_degree <= 0:
        return _empty_graph(n)

    rng = np.random.default_rng(seed)
    # Collected as arrays of (i, j) pairs per outer iteration rather than
    # inserted into a Python set one at a time -- each i produces at most
    # n-1-i pairs and, since j always comes from i+1.., no (i, j) pair can
    # recur across iterations, so no dedup is needed, only a final sort.
    from_chunks: list[np.ndarray] = []
    to_chunks: list[np.ndarray] = []
    for i in range(n - 1):
        if deg[i] <= 0:
            continue
        probs = np.clip((deg[i] * deg[i + 1 :]) / total_degree, 0.0, 1.0)
        if probs.size == 0:
            continue
        offsets = np.flatnonzero(rng.random(probs.size) < probs)
        if offsets.size == 0:
            continue
        from_chunks.append(np.full(offsets.size, i, dtype=np.uint32))
        to_chunks.append((i + 1 + offsets).astype(np.uint32))

    if not from_chunks:
        return _empty_graph(n)
    edge_from = np.concatenate(from_chunks)
    edge_to = np.concatenate(to_chunks)
    order = np.lexsort((edge_to, edge_from))
    return NetworkGraph(node_count=n, edge_from=edge_from[order], edge_to=edge_to[order])


def random_persistence(
    graph: NetworkGraph,
    degrees: np.ndarray,
    *,
    time_steps: int,
    seed: int,
) -> np.ndarray:
    """Synthetic per-edge persistence for a :func:`degree_preserving_random_graph`,
    sampled as `Binomial(time_steps, p) / time_steps` with the same
    degree-product probability used to generate the graph's edges.
    """
    if time_steps <= 0 or graph.edge_count == 0:
        return np.asarray([], dtype=float)
    deg = np.asarray(degrees, dtype=float)
    total_degree = float(deg.sum())
    if total_degree <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    probs = np.clip((deg[graph.edge_from] * deg[graph.edge_to]) / total_degree, 0.0, 1.0)
    return rng.binomial(time_steps, probs) / time_steps


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Summary stats (count/mean/median/std/p10/p90) of a metric distribution,
    ignoring non-finite values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def infer_social_ties(
    graph: NetworkGraph,
    edge_persistence: np.ndarray,
    *,
    regularity_threshold: float,
    overlap_threshold: float = 0.0,
    random_chance_probability: float = 1e-3,
    random_baseline_samples: int = 10_000,
    seed: int = 42,
) -> NetworkGraph:
    """Infer genuine social ties from a co-presence graph (Fournet &
    Barrat-style random-graph-baseline significance test).

    A co-presence pair is promoted to an inferred social-tie edge if both:

    1. ``regularity = edge_persistence >= regularity_threshold`` (the pair
       co-occurs often enough to not be a one-off encounter).
    2. Their topological overlap (Jaccard similarity of neighbor sets) is at
       or above ``max(overlap_threshold, random_baseline)``, where
       ``random_baseline`` is the ``(1 - random_chance_probability)``-quantile
       of topological overlap sampled over random node pairs in the same
       graph -- i.e. their neighborhood similarity is higher than random
       chance alone would produce at the requested significance level.

    This method has been validated against real observed co-presence data:
    it gives a plausible ~103 mean friendship degree on a real dense urban
    dataset at ``random_chance_probability=1e-3``, though it can degenerate
    on coarser fixed-grid location data (the random baseline saturates when
    co-presence is too dense to separate "social" from "random" at that
    spatial resolution) -- inspect the output graph's mean degree when using
    this on a new dataset.

    Parameters
    ----------
    graph:
        A co-presence graph, e.g. from :func:`co_presence_graph_from_visits`.
    edge_persistence:
        Per-edge persistence (same order as ``graph.edge_from``/``edge_to``),
        e.g. from :func:`co_presence_graph_from_visits`.
    regularity_threshold:
        Minimum persistence for a pair to be considered for promotion.
    overlap_threshold:
        A floor on topological overlap, applied in addition to (not instead
        of) the random-baseline significance threshold.
    random_chance_probability:
        Significance level for the random-baseline overlap threshold
        (smaller = stricter; only overlaps in the top
        ``random_chance_probability`` fraction of the random-pair null
        distribution pass).
    random_baseline_samples:
        Number of random node pairs sampled to estimate the null
        distribution.
    seed:
        Seed for the random-baseline sampler; deterministic given the same
        seed and graph.

    Returns
    -------
    NetworkGraph
        A new, generally sparser graph containing only the promoted edges.

    Examples
    --------
    >>> import numpy as np
    >>> from fastmob.social import NetworkGraph, infer_social_ties
    >>> graph = NetworkGraph(
    ...     node_count=4,
    ...     edge_from=np.array([0, 1, 2], dtype=np.uint32),
    ...     edge_to=np.array([1, 2, 3], dtype=np.uint32),
    ... )
    >>> persistence = np.array([0.9, 0.9, 0.1])
    >>> inferred = infer_social_ties(graph, persistence, regularity_threshold=0.5, overlap_threshold=0.0)
    >>> inferred.edge_count <= graph.edge_count
    True
    """
    from fastmob._core import graph_metrics as _graph_metrics_core
    from fastmob._core import random_baseline_overlap_threshold as _random_baseline_overlap_threshold_core

    if graph.edge_count == 0:
        return _empty_graph(graph.node_count)

    _clustering, overlap = _graph_metrics_core(graph.node_count, graph.edge_from, graph.edge_to)
    random_baseline = _random_baseline_overlap_threshold_core(
        graph.node_count,
        graph.edge_from,
        graph.edge_to,
        random_baseline_samples,
        random_chance_probability,
        seed,
    )
    effective_overlap_threshold = max(overlap_threshold, random_baseline)

    persistence = np.asarray(edge_persistence, dtype=float)
    promote = (persistence >= regularity_threshold) & (overlap >= effective_overlap_threshold)

    return NetworkGraph(
        node_count=graph.node_count,
        edge_from=graph.edge_from[promote],
        edge_to=graph.edge_to[promote],
    )


NetworkGraph.__module__ = "fastmob.social"
for _public_function in (
    clustering_coefficients,
    co_presence_graph_from_visits,
    co_presence_graph_from_staypoints,
    degree_preserving_random_graph,
    distribution_summary,
    graph_from_edges,
    infer_social_ties,
    random_persistence,
    topological_overlap,
):
    _public_function.__module__ = "fastmob.social"
