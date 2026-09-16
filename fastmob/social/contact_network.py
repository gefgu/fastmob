"""Raw co-presence graphs and generic contact-network diagnostics.

These utilities construct who shared a location on a calendar day.  They do
not run RECAST or infer social relationship classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from fastmob.measures.evaluation.metrics import wasserstein_distance
from .recast import RecastTemporalGraph, temporal_graph_from_staypoints


@dataclass(frozen=True)
class NetworkGraph:
    """An undirected graph stored as sorted, dense integer edge arrays."""

    node_count: int
    edge_from: np.ndarray
    edge_to: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(self.edge_from.size)

    @property
    def edges(self) -> set[tuple[int, int]]:
        return set(zip(self.edge_from.tolist(), self.edge_to.tolist()))

    def degrees(self) -> np.ndarray:
        if not self.edge_count:
            return np.zeros(self.node_count, dtype=np.int64)
        return np.bincount(
            np.concatenate((self.edge_from, self.edge_to)), minlength=self.node_count
        )


@dataclass(frozen=True)
class ContactNetworkResult:
    """Daily RECAST event graphs plus their raw aggregate contact network."""

    temporal_graph: RecastTemporalGraph
    graph: NetworkGraph
    edge_persistence: np.ndarray

    @property
    def time_steps(self) -> int:
        return self.temporal_graph.time_steps


def _empty_graph(node_count: int) -> NetworkGraph:
    empty = np.empty(0, dtype=np.uint32)
    return NetworkGraph(node_count, empty, empty)


def graph_from_edges(node_count: int, edges: Iterable[tuple[int, int]]) -> NetworkGraph:
    """Construct a normalized graph from an iterable of undirected edges."""
    normalized = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edges
        if int(u) != int(v) and 0 <= int(u) < node_count and 0 <= int(v) < node_count
    }
    if not normalized:
        return _empty_graph(node_count)
    array = np.asarray(sorted(normalized), dtype=np.uint32)
    return NetworkGraph(node_count, array[:, 0], array[:, 1])


def co_presence_graph_from_staypoints(
    staypoints: Any,
    locations: Any,
    *,
    min_minutes_for_encounter: int = 5,
) -> ContactNetworkResult:
    """Build raw contacts from RECAST's daily interval-overlap event graphs.

    The same globally located staypoints, strict interval-overlap rule, and
    minimum encounter duration as RECAST are used. The daily event graphs are
    returned alongside their aggregate graph and per-edge persistence. No
    random thresholds or relationship classes are inferred.
    """
    temporal_graph = temporal_graph_from_staypoints(
        staypoints, locations, min_minutes_for_encounter=min_minutes_for_encounter
    )
    from fastmob._core import recast_aggregate_event_graphs

    edge_from, edge_to, persistence = recast_aggregate_event_graphs(
        temporal_graph.edge_offsets, temporal_graph.edge_from, temporal_graph.edge_to
    )
    graph = NetworkGraph(temporal_graph.node_count, np.asarray(edge_from), np.asarray(edge_to))
    return ContactNetworkResult(temporal_graph, graph, np.asarray(persistence))


def _metrics(graph: NetworkGraph) -> tuple[np.ndarray, np.ndarray]:
    from fastmob._core import contact_graph_metrics

    return tuple(np.asarray(x) for x in contact_graph_metrics(graph.node_count, graph.edge_from, graph.edge_to))


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    """Return one clustering coefficient per node."""
    return _metrics(graph)[0]


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    """Return Jaccard neighborhood overlap aligned with graph edges."""
    return _metrics(graph)[1]


def degree_preserving_random_graph(degrees: np.ndarray, *, seed: int = 42) -> NetworkGraph:
    """Sample a Chung-Lu graph whose degrees match in expectation."""
    degree = np.asarray(degrees, dtype=float)
    total = float(degree.sum())
    if degree.size < 2 or total <= 0:
        return _empty_graph(len(degree))
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for source in range(len(degree) - 1):
        probability = np.clip(degree[source] * degree[source + 1 :] / total, 0.0, 1.0)
        targets = np.flatnonzero(rng.random(probability.size) < probability)
        if targets.size:
            chunks.append(np.column_stack((np.full(targets.size, source), targets + source + 1)))
    return graph_from_edges(len(degree), np.vstack(chunks) if chunks else [])


def random_persistence(graph: NetworkGraph, degrees: np.ndarray, *, time_steps: int, seed: int) -> np.ndarray:
    """Sample persistence for a Chung-Lu baseline graph."""
    degree = np.asarray(degrees, dtype=float)
    total = float(degree.sum())
    if not graph.edge_count or time_steps <= 0 or total <= 0:
        return np.asarray([], dtype=float)
    probability = np.clip(degree[graph.edge_from] * degree[graph.edge_to] / total, 0.0, 1.0)
    return np.random.default_rng(seed).binomial(time_steps, probability) / time_steps


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Return stable summary statistics after dropping non-finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ("mean", "median", "std", "p10", "p90")} | {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def safe_wasserstein(left: np.ndarray, right: np.ndarray) -> float | None:
    value = wasserstein_distance(left, right)
    return None if math.isnan(value) else value


for _item in (
    ContactNetworkResult,
    NetworkGraph,
    graph_from_edges,
    co_presence_graph_from_staypoints,
    clustering_coefficients, topological_overlap, degree_preserving_random_graph, random_persistence,
    distribution_summary, safe_wasserstein,
):
    _item.__module__ = "fastmob.social"
