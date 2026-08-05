"""Arrow-backed RECAST classification and paper-validation diagnostics.

RECAST uses daily UTC event graphs.  This is deliberate: social encounters
are evaluated as recurring daily routines, rather than using a millisecond
window parameter.  ``min_minutes_for_encounter`` is only the minimum strict
co-presence duration required to create a daily contact edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob.core import Locations, Staypoints
from fastmob.utils._common import _factorize_arrow_values


class RecastClass(IntEnum):
    """Relationship classes defined by RECAST Algorithm 1."""

    FRIENDS = 0
    BRIDGES = 1
    ACQUAINTANCES = 2
    RANDOM = 3


@dataclass(frozen=True)
class RecastEventGraph:
    """One Arrow-backed, undirected event graph with integer node ids."""

    node_count: int
    edge_from: pa.Array
    edge_to: pa.Array


@dataclass(frozen=True)
class RecastTemporalGraph:
    """Packed Arrow event graphs; offsets delimit the edges for each day."""

    node_count: int
    node_ids: pa.Array
    window_starts_ms: pa.Array
    edge_offsets: pa.Array
    edge_from: pa.Array
    edge_to: pa.Array

    @property
    def time_steps(self) -> int:
        return len(self.window_starts_ms)

    def event(self, index: int) -> RecastEventGraph:
        if not 0 <= index < self.time_steps:
            raise IndexError("event index out of range")
        start = self.edge_offsets[index].as_py()
        end = self.edge_offsets[index + 1].as_py()
        return RecastEventGraph(
            self.node_count, self.edge_from.slice(start, end - start), self.edge_to.slice(start, end - start)
        )


@dataclass(frozen=True)
class RecastResult:
    """Classification arrays aligned one-for-one with aggregate observed edges."""

    source_users: pa.Array
    target_users: pa.Array
    edge_persistence: pa.Array
    topological_overlap: pa.Array
    classes: pa.Array
    persistence_threshold: float
    overlap_threshold: float
    p_rnd: float
    random_replicates: int
    seed: int
    min_minutes_for_encounter: int
    time_steps: int

    @property
    def class_names(self) -> pa.Array:
        return pc.take(pa.array(["Friends", "Bridges", "Acquaintances", "Random"]), self.classes)

    def mask(self, relationship: RecastClass) -> pa.Array:
        return pc.equal(self.classes, pa.scalar(int(relationship), pa.uint8()))


@dataclass(frozen=True)
class RecastClusteringComparison:
    """Cumulative clustering coefficients for observed G_t and T-RND G^R_t."""

    window_starts_ms: pa.Array
    observed: pa.Array
    random_mean: pa.Array
    random_std: pa.Array


@dataclass(frozen=True)
class RecastValidation:
    """Section 4/5 RECAST validation artifacts, all in Arrow arrays.

    ``*_null`` arrays are the pooled RND/T-RND distributions used to infer
    thresholds and plot the paper's CCDF diagnostics.  ``full_clustering``
    corresponds to Figure 3; ``random_only_clustering`` corresponds to Figure
    8 after retaining only edges classified as ``Random``.
    """

    classification: RecastResult
    graph: RecastTemporalGraph
    persistence_observed: pa.Array
    overlap_observed: pa.Array
    persistence_null: pa.Array
    overlap_null: pa.Array
    full_clustering: RecastClusteringComparison
    random_only_clustering: RecastClusteringComparison


@dataclass(frozen=True)
class _PreparedStaypoints:
    node_ids: pa.Array
    users: pa.Array
    locations: pa.Array
    starts: pa.Array
    ends: pa.Array

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def _validate_options(min_minutes_for_encounter: int, p_rnd: float, random_replicates: int) -> int:
    if (
        not isinstance(min_minutes_for_encounter, int)
        or isinstance(min_minutes_for_encounter, bool)
        or min_minutes_for_encounter <= 0
    ):
        raise ValueError("min_minutes_for_encounter must be a positive integer")
    if not 0.0 <= p_rnd <= 1.0:
        raise ValueError("p_rnd must be between 0 and 1")
    if isinstance(random_replicates, bool) or random_replicates < 1:
        raise ValueError("random_replicates must be at least 1")
    return min_minutes_for_encounter * 60_000


def _prepare(staypoints: Staypoints, locations: Locations) -> _PreparedStaypoints:
    if not isinstance(staypoints, Staypoints):
        raise TypeError("RECAST requires a Staypoints instance")
    if not isinstance(locations, Locations) or locations.scope != "global":
        raise ValueError("RECAST requires global Locations")
    assigned = staypoints.associate_global_locations(locations)
    if assigned.finished_at_col is None:
        raise ValueError("RECAST requires Staypoints with finished_at timestamps")
    uid_col, start_col, end_col = assigned.uid_col, assigned.started_at_col, assigned.finished_at_col
    work = (
        nw.from_native(assigned.df, eager_only=True).select([uid_col, "location_id", start_col, end_col]).drop_nulls()
    )
    if len(work) == 0:
        return _PreparedStaypoints(
            pa.array([], type=pa.null()),
            pa.array([], type=pa.uint32()),
            pa.array([], type=pa.uint32()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
        )
    uid_input = work.get_column(uid_col).to_arrow()
    uid_codes, uid_representatives = _factorize_arrow_values(uid_input, sort=True)
    location_codes, _ = _factorize_arrow_values(work.get_column("location_id").to_arrow(), sort=True)
    return _PreparedStaypoints(
        pc.take(pa.array(uid_input), uid_representatives),
        pc.cast(uid_codes, pa.uint32()),
        pc.cast(location_codes, pa.uint32()),
        work.get_column(start_col).dt.timestamp("ms").cast(nw.Int64).to_arrow(),
        work.get_column(end_col).dt.timestamp("ms").cast(nw.Int64).to_arrow(),
    )


def _result(
    prepared: _PreparedStaypoints,
    values: tuple,
    *,
    min_minutes_for_encounter: int,
    p_rnd: float,
    random_replicates: int,
    seed: int,
) -> RecastResult:
    edge_from, edge_to, persistence, overlap, classes, p_threshold, o_threshold, time_steps = values
    edge_from, edge_to, persistence, overlap, classes = map(
        pa.array, (edge_from, edge_to, persistence, overlap, classes)
    )
    return RecastResult(
        pc.take(prepared.node_ids, edge_from),
        pc.take(prepared.node_ids, edge_to),
        persistence,
        overlap,
        classes,
        float(p_threshold),
        float(o_threshold),
        p_rnd,
        random_replicates,
        seed,
        min_minutes_for_encounter,
        int(time_steps),
    )


def recast_from_staypoints(
    staypoints: Staypoints,
    locations: Locations,
    *,
    min_minutes_for_encounter: int = 5,
    p_rnd: float = 1e-3,
    random_replicates: int = 5,
    seed: int = 42,
) -> RecastResult:
    """Classify contacts into Friends, Bridges, Acquaintances, and Random.

    The random null model is the paper's degree-product RND, independently
    sampled per daily event graph by T-RND.  It preserves snapshot degrees in
    expectation, not exactly in each realization.
    """
    min_contact_ms = _validate_options(min_minutes_for_encounter, p_rnd, random_replicates)
    prepared = _prepare(staypoints, locations)
    if prepared.node_count == 0:
        empty = pa.array([], type=pa.null())
        return RecastResult(
            empty,
            empty,
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.uint8()),
            1.0,
            1.0,
            p_rnd,
            random_replicates,
            seed,
            min_minutes_for_encounter,
            0,
        )
    from fastmob._core import recast_classify

    values = recast_classify(
        prepared.node_count,
        prepared.users,
        prepared.locations,
        prepared.starts,
        prepared.ends,
        min_contact_ms,
        p_rnd,
        random_replicates,
        seed,
    )
    return _result(
        prepared,
        values,
        min_minutes_for_encounter=min_minutes_for_encounter,
        p_rnd=p_rnd,
        random_replicates=random_replicates,
        seed=seed,
    )


def temporal_graph_from_staypoints(
    staypoints: Staypoints, locations: Locations, *, min_minutes_for_encounter: int = 5
) -> RecastTemporalGraph:
    """Construct the daily RECAST event graphs without running classification."""
    min_contact_ms = _validate_options(min_minutes_for_encounter, 1e-3, 1)
    prepared = _prepare(staypoints, locations)
    if prepared.node_count == 0:
        return RecastTemporalGraph(
            0,
            prepared.node_ids,
            pa.array([], type=pa.int64()),
            pa.array([0], type=pa.uint64()),
            pa.array([], type=pa.uint32()),
            pa.array([], type=pa.uint32()),
        )
    from fastmob._core import recast_event_graphs

    windows, offsets, edge_from, edge_to = recast_event_graphs(
        prepared.users, prepared.locations, prepared.starts, prepared.ends, min_contact_ms
    )
    return RecastTemporalGraph(
        prepared.node_count,
        prepared.node_ids,
        pa.array(windows),
        pa.array(offsets),
        pa.array(edge_from),
        pa.array(edge_to),
    )


def rnd(event_graph: RecastEventGraph, *, seed: int = 42) -> RecastEventGraph:
    """Generate one paper-RND graph from a RECAST event graph."""
    if not isinstance(event_graph, RecastEventGraph):
        raise TypeError("rnd requires a RecastEventGraph")
    from fastmob._core import recast_rnd

    edge_from, edge_to = recast_rnd(event_graph.node_count, event_graph.edge_from, event_graph.edge_to, seed)
    return RecastEventGraph(event_graph.node_count, pa.array(edge_from), pa.array(edge_to))


def t_rnd(
    temporal_graph: RecastTemporalGraph, *, random_replicates: int = 5, seed: int = 42
) -> list[RecastTemporalGraph]:
    """Generate independent T-RND replicas for every daily event graph."""
    if not isinstance(temporal_graph, RecastTemporalGraph):
        raise TypeError("t_rnd requires a RecastTemporalGraph")
    _validate_options(1, 1e-3, random_replicates)
    from fastmob._core import recast_t_rnd

    return [
        RecastTemporalGraph(
            temporal_graph.node_count,
            temporal_graph.node_ids,
            *map(
                pa.array,
                recast_t_rnd(
                    temporal_graph.node_count,
                    temporal_graph.window_starts_ms,
                    temporal_graph.edge_offsets,
                    temporal_graph.edge_from,
                    temporal_graph.edge_to,
                    replica,
                    seed,
                ),
            ),
        )
        for replica in range(random_replicates)
    ]


def validate_recast_from_staypoints(
    staypoints: Staypoints,
    locations: Locations,
    *,
    min_minutes_for_encounter: int = 5,
    p_rnd: float = 1e-3,
    random_replicates: int = 5,
    seed: int = 42,
) -> RecastValidation:
    """Return the classifier plus Figure 3/4/8 validation diagnostics.

    This library-scope API intentionally stops at validating RECAST's temporal
    graph model; the paper's opportunistic-routing and Facebook studies remain
    application-level evaluations.
    """
    min_contact_ms = _validate_options(min_minutes_for_encounter, p_rnd, random_replicates)
    prepared = _prepare(staypoints, locations)
    if prepared.node_count == 0:
        result = recast_from_staypoints(
            staypoints,
            locations,
            min_minutes_for_encounter=min_minutes_for_encounter,
            p_rnd=p_rnd,
            random_replicates=random_replicates,
            seed=seed,
        )
        graph = temporal_graph_from_staypoints(
            staypoints, locations, min_minutes_for_encounter=min_minutes_for_encounter
        )
        empty = pa.array([], type=pa.float64())
        comparison = RecastClusteringComparison(graph.window_starts_ms, empty, empty, empty)
        return RecastValidation(result, graph, empty, empty, empty, empty, comparison, comparison)
    from fastmob._core import recast_validate

    values = recast_validate(
        prepared.node_count,
        prepared.users,
        prepared.locations,
        prepared.starts,
        prepared.ends,
        min_contact_ms,
        p_rnd,
        random_replicates,
        seed,
    )
    result = _result(
        prepared,
        values[:8],
        min_minutes_for_encounter=min_minutes_for_encounter,
        p_rnd=p_rnd,
        random_replicates=random_replicates,
        seed=seed,
    )
    graph = RecastTemporalGraph(prepared.node_count, prepared.node_ids, *map(pa.array, values[18:22]))
    full = RecastClusteringComparison(
        graph.window_starts_ms, pa.array(values[12]), pa.array(values[13]), pa.array(values[14])
    )
    random_only = RecastClusteringComparison(
        graph.window_starts_ms, pa.array(values[15]), pa.array(values[16]), pa.array(values[17])
    )
    return RecastValidation(
        result,
        graph,
        pa.array(values[8]),
        pa.array(values[9]),
        pa.array(values[10]),
        pa.array(values[11]),
        full,
        random_only,
    )


for _item in (
    RecastClass,
    RecastEventGraph,
    RecastTemporalGraph,
    RecastResult,
    RecastClusteringComparison,
    RecastValidation,
    recast_from_staypoints,
    temporal_graph_from_staypoints,
    rnd,
    t_rnd,
    validate_recast_from_staypoints,
):
    _item.__module__ = "fastmob.social"
