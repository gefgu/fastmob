"""Raw co-presence graphs and generic contact-network diagnostics.

These utilities construct who shared a location on a calendar day.  They do
not run RECAST or infer social relationship classes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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
        """Return the number of contacts of each node, as an integer array of length ``node_count``."""
        if not self.edge_count:
            return np.zeros(self.node_count, dtype=np.int64)
        return np.bincount(np.concatenate((self.edge_from, self.edge_to)), minlength=self.node_count)


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
    """Build a normalized undirected graph from a list of edges.

    Parameters
    ----------
    node_count : int
        Number of nodes; nodes are the integers ``0`` to ``node_count - 1``.
    edges : iterable of (int, int)
        Undirected edges. Self-loops are dropped and duplicate edges (in either direction) are merged.

    Returns
    -------
    NetworkGraph
        The graph, with ``edge_from``, ``edge_to``, ``node_count`` and ``edge_count`` and the method ``degrees()``.
    """
    normalized = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edges
        if int(u) != int(v) and 0 <= int(u) < node_count and 0 <= int(v) < node_count
    }
    if not normalized:
        return _empty_graph(node_count)
    array = np.asarray(sorted(normalized), dtype=np.uint32)
    return NetworkGraph(node_count, np.ascontiguousarray(array[:, 0]), np.ascontiguousarray(array[:, 1]))


def co_presence_graph_from_staypoints(
    staypoints: Any,
    locations: Any,
    *,
    min_minutes_for_encounter: int = 5,
) -> ContactNetworkResult:
    """Build the aggregate contact network from the co-presence of users at the same location.

    Parameters
    ----------
    staypoints : Staypoints
        Stays that carry a global ``location_id``.
    locations : Locations
        The global location catalogue.
    min_minutes_for_encounter : int, optional
        Minimum time overlap, in minutes, for two users to count as in contact on a day. Default 5.

    Returns
    -------
    ContactNetworkResult
        ``graph`` (undirected, nodes are users), ``temporal_graph`` (daily graphs), ``time_steps`` (number of
        days) and ``edge_persistence`` (share of days on which each contact occurred, 0 to 1).
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


def clustering_coefficients(graph: NetworkGraph) -> np.ndarray:
    """Return the local clustering coefficient of every node.

    For a node with at least two neighbours it is the share of pairs of its neighbours that are also
    connected to each other (0 to 1); for a node with fewer than two neighbours it is 0.

    Parameters
    ----------
    graph : NetworkGraph
        An undirected graph, for example ``contact_network.graph``.

    Returns
    -------
    numpy.ndarray
        One value per node, in node order.
    """
    from fastmob._core import recast_local_clustering_coefficients

    return np.asarray(recast_local_clustering_coefficients(graph.node_count, graph.edge_from, graph.edge_to))


def topological_overlap(graph: NetworkGraph) -> np.ndarray:
    """Return the neighbourhood overlap of every edge.

    For an edge between nodes ``u`` and ``v`` it is the Jaccard overlap of their neighbourhoods: the number
    of neighbours they share divided by the size of the union of their neighbourhoods (0 to 1). High values
    mean the two people have many common contacts.

    Parameters
    ----------
    graph : NetworkGraph
        An undirected graph.

    Returns
    -------
    numpy.ndarray
        One value per edge, in the order of ``graph.edge_from`` and ``graph.edge_to``.
    """
    from fastmob._core import recast_topological_overlaps

    return np.asarray(recast_topological_overlaps(graph.node_count, graph.edge_from, graph.edge_to))


def degree_preserving_random_graph(degrees: np.ndarray, *, seed: int = 42) -> NetworkGraph:
    """Sample a random graph whose node degrees match the given degrees in expectation (Chung-Lu model).

    Parameters
    ----------
    degrees : numpy.ndarray
        Target degree of each node; its length is the number of nodes.
    seed : int, optional
        Random seed. Default 42.

    Returns
    -------
    NetworkGraph
        A random undirected graph. Its degrees equal ``degrees`` only on average, so the number of edges
        differs slightly from one sample to the next.
    """
    degree = np.asarray(degrees)
    if degree.ndim != 1:
        raise ValueError("degrees must be a one-dimensional array")
    if not np.issubdtype(degree.dtype, np.number):
        raise ValueError("degrees must be numeric")
    degree_float = degree.astype(float, copy=False)
    if (
        not np.all(np.isfinite(degree_float))
        or np.any(degree_float < 0)
        or np.any(degree_float != np.floor(degree_float))
    ):
        raise ValueError("degrees must be finite, non-negative integers")
    if np.any(degree_float >= degree.size):
        raise ValueError("each degree must be smaller than the node count")
    if degree.size < 2 or not degree_float.sum():
        return _empty_graph(len(degree))
    if np.any(degree_float > np.iinfo(np.uint64).max):
        raise ValueError("degrees exceed the supported range")
    from fastmob._core import recast_expected_degree_graph

    edge_from, edge_to = recast_expected_degree_graph(np.ascontiguousarray(degree_float, dtype=np.uint64), seed)
    return NetworkGraph(len(degree), np.asarray(edge_from), np.asarray(edge_to))


def random_persistence(graph: NetworkGraph, degrees: np.ndarray, *, time_steps: int, seed: int) -> np.ndarray:
    """Sample the persistence that edges of a degree-preserving random baseline would have.

    Parameters
    ----------
    graph : NetworkGraph
        The observed graph, whose edges define how many values are drawn.
    degrees : numpy.ndarray
        Degree of each node used by the Chung-Lu baseline.
    time_steps : int
        Number of time steps (days) over which persistence is measured.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray
        One persistence value (share of time steps in which the edge exists, 0 to 1) per sampled edge.
    """
    degree = np.asarray(degrees, dtype=float)
    total = float(degree.sum())
    if not graph.edge_count or time_steps <= 0 or total <= 0:
        return np.asarray([], dtype=float)
    probability = np.clip(degree[graph.edge_from] * degree[graph.edge_to] / total, 0.0, 1.0)
    return np.random.default_rng(seed).binomial(time_steps, probability) / time_steps


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Summarize a sample with stable statistics after dropping non-finite values.

    Parameters
    ----------
    values : numpy.ndarray
        The sample.

    Returns
    -------
    dict
        ``count``, ``mean``, ``median``, ``std``, ``p10`` and ``p90``; the statistics are ``None`` when no
        finite value remains.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ("mean", "median", "std", "p10", "p90")} | {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def safe_wasserstein(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return the 1-D Wasserstein distance between two samples, or ``None`` when it is undefined.

    Parameters
    ----------
    left, right : numpy.ndarray
        The two samples.

    Returns
    -------
    float or None
        The distance in the units of the values, or ``None`` if it is NaN (for example for an empty sample).
    """
    value = wasserstein_distance(left, right)
    return None if math.isnan(value) else value


for _item in (
    ContactNetworkResult,
    NetworkGraph,
    graph_from_edges,
    co_presence_graph_from_staypoints,
    clustering_coefficients,
    topological_overlap,
    degree_preserving_random_graph,
    random_persistence,
    distribution_summary,
    safe_wasserstein,
):
    _item.__module__ = "fastmob.social"
