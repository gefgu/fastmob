"""Arrow-backed, faithful RECAST classification of temporal contacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob.core import Locations, Staypoints
from fastmob.utils._common import _factorize_arrow_values


class RecastClass(IntEnum):
    """RECAST relationship classes."""

    FRIENDS = 0
    BRIDGES = 1
    ACQUAINTANCES = 2
    RANDOM = 3


@dataclass(frozen=True)
class RecastResult:
    """Arrow arrays aligned one-for-one with observed aggregate edges."""

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


def recast_from_staypoints(
    staypoints: Staypoints,
    locations: Locations,
    *,
    min_minutes_for_encounter: int = 5,
    p_rnd: float = 1e-3,
    random_replicates: int = 5,
    seed: int = 42,
    swaps_per_edge: int = 10,
) -> RecastResult:
    """Run RECAST over globally located staypoints using Arrow-only columns.

    RECAST uses UTC calendar days as temporal snapshots, as in the paper's
    human-routine evaluation. A daily contact requires users to be at the
    same global location with strict interval overlap lasting at least
    ``min_minutes_for_encounter``. T-RND randomizes every daily event graph
    through exact degree-preserving double-edge swaps before threshold
    inference.
    """
    if not isinstance(staypoints, Staypoints):
        raise TypeError("recast_from_staypoints requires a Staypoints instance")
    if not isinstance(locations, Locations) or locations.scope != "global":
        raise ValueError("recast_from_staypoints requires global Locations")
    if not 0.0 <= p_rnd <= 1.0:
        raise ValueError("p_rnd must be between 0 and 1")
    if isinstance(random_replicates, bool) or random_replicates < 1:
        raise ValueError("random_replicates must be at least 1")
    if isinstance(swaps_per_edge, bool) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be non-negative")
    if not isinstance(min_minutes_for_encounter, int) or isinstance(min_minutes_for_encounter, bool) or min_minutes_for_encounter <= 0:
        raise ValueError("min_minutes_for_encounter must be a positive integer")
    min_contact_ms = min_minutes_for_encounter * 60_000

    assigned = staypoints.associate_global_locations(locations)
    end_col = assigned.finished_at_col
    if end_col is None:
        raise ValueError("RECAST requires Staypoints with finished_at timestamps")
    uid_col, start_col = assigned.uid_col, assigned.started_at_col
    df = nw.from_native(assigned.df, eager_only=True)
    work = df.select([uid_col, "location_id", start_col, end_col]).drop_nulls()
    empty = pa.array([], type=pa.null())
    if len(work) == 0:
        return RecastResult(empty, empty, pa.array([], type=pa.float64()), pa.array([], type=pa.float64()), pa.array([], type=pa.uint8()), 1.0, 1.0, p_rnd, random_replicates, seed, min_minutes_for_encounter, 0)

    uid_input = work.get_column(uid_col).to_arrow()
    uid_codes, uid_representatives = _factorize_arrow_values(uid_input, sort=True)
    location_codes, _ = _factorize_arrow_values(work.get_column("location_id").to_arrow(), sort=True)
    uid_codes = pc.cast(uid_codes, pa.uint32())
    location_codes = pc.cast(location_codes, pa.uint32())
    starts = work.get_column(start_col).dt.timestamp("ms").cast(nw.Int64).to_arrow()
    ends = work.get_column(end_col).dt.timestamp("ms").cast(nw.Int64).to_arrow()
    uid_labels = pc.take(pa.array(uid_input), uid_representatives)

    from fastmob._core import recast_classify

    edge_from, edge_to, persistence, overlap, classes, persistence_threshold, overlap_threshold, time_steps = recast_classify(
        len(uid_labels), uid_codes, location_codes, starts, ends, min_contact_ms, p_rnd, random_replicates, seed, swaps_per_edge
    )
    edge_from, edge_to = pa.array(edge_from), pa.array(edge_to)
    persistence, overlap, classes = pa.array(persistence), pa.array(overlap), pa.array(classes)
    return RecastResult(
        pc.take(uid_labels, edge_from), pc.take(uid_labels, edge_to), persistence, overlap, classes,
        float(persistence_threshold), float(overlap_threshold), p_rnd, random_replicates, seed, min_minutes_for_encounter, int(time_steps),
    )


RecastClass.__module__ = "fastmob.social"
RecastResult.__module__ = "fastmob.social"
recast_from_staypoints.__module__ = "fastmob.social"
