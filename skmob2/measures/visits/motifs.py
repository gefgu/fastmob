"""Mobility motif classification without networkx.

Implements the directed-graph motif library from Pappalardo et al. using
plain Python adjacency-list representations for full independence from
the networkx library.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import narwhals as nw

from skmob2._core import canonical_adjacency_form as _canonical_adjacency_form_rust

from .._common import (
    _pick_existing_column,
    USER_ID_CANDIDATES,
    LOCATION_CANDIDATES,
    PURPOSE_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    DURATION_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Internal graph representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DiGraph:
    """Minimal directed graph for motif matching.

    Parameters
    ----------
    n_nodes:
        Total node count (nodes are identified by integers 0..n_nodes-1).
    edges:
        Frozen set of ``(u, v)`` directed edge pairs.
    """

    n_nodes: int
    edges: frozenset


def _degree_sequence(n_nodes: int, edges: frozenset) -> tuple:
    """Return sorted (in_degree, out_degree) pairs for all nodes.

    Used as a fast prefilter before the full permutation search: two graphs
    with different degree sequences cannot be isomorphic.

    Parameters
    ----------
    n_nodes:
        Number of nodes (nodes are labeled 0..n_nodes-1).
    edges:
        Directed edge set as a frozenset of ``(u, v)`` pairs.

    Returns
    -------
    tuple
        Sorted tuple of ``(in_degree, out_degree)`` pairs, one per node.
    """
    in_deg = [0] * n_nodes
    out_deg = [0] * n_nodes
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1
    return tuple(sorted(zip(in_deg, out_deg)))


def _canonical_adjacency_form(n_nodes: int, edges: frozenset) -> str:
    """Return the canonical adjacency-matrix string for a directed graph.

    Delegates to the Rust kernel ``_core.canonical_adjacency_form`` which
    enumerates all n! permutations of node labels, builds the binary
    adjacency matrix for each permutation, and returns the maximum value
    as a zero-padded big-endian binary string of length n*n.

    Two graphs are isomorphic if and only if their canonical forms are equal.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edges:
        Directed edge set as a frozenset of ``(u, v)`` pairs.

    Returns
    -------
    str
        Zero-padded big-endian binary string of length ``n_nodes ** 2``.
        For example, the 2-node bidirectional graph returns ``"0110"``.

    Examples
    --------
    >>> _canonical_adjacency_form(1, frozenset())
    '0'
    >>> _canonical_adjacency_form(2, frozenset({(0, 1), (1, 0)}))
    '0110'
    """
    return _canonical_adjacency_form_rust(n_nodes, list(edges))


def _is_isomorphic(g1: _DiGraph, g2: _DiGraph) -> bool:
    """Check whether two directed graphs are isomorphic via canonical forms.

    Uses a degree-sequence prefilter for fast rejection before computing
    canonical adjacency-matrix strings.  Two graphs are isomorphic if and
    only if their canonical forms (as returned by
    ``_canonical_adjacency_form``) are equal.

    Works for up to ~8 nodes (8! = 40 320 permutations — fast enough for
    the static motif library which has at most 6 nodes).

    Parameters
    ----------
    g1, g2:
        The two directed graphs to compare.

    Returns
    -------
    bool
        True if g1 and g2 are isomorphic.
    """
    if g1.n_nodes != g2.n_nodes or len(g1.edges) != len(g2.edges):
        return False

    # Fast rejection: different degree sequences → cannot be isomorphic
    if _degree_sequence(g1.n_nodes, g1.edges) != _degree_sequence(g2.n_nodes, g2.edges):
        return False

    return _canonical_adjacency_form(g1.n_nodes, g1.edges) == _canonical_adjacency_form(
        g2.n_nodes, g2.edges
    )


# ---------------------------------------------------------------------------
# Static motif library (17 canonical mobility patterns)
# ---------------------------------------------------------------------------


def get_motif_library() -> dict[int, _DiGraph]:
    """Return the 17 canonical directed mobility motifs.

    Returns
    -------
    dict[int, _DiGraph]
        Mapping of integer motif ID (1–17) to its canonical ``_DiGraph``.
    """
    library: dict[int, _DiGraph] = {}
    index = 1

    # --- Size N=1 ---
    # ID 1: Single location (Home only)
    library[index] = _DiGraph(n_nodes=1, edges=frozenset())
    index += 1

    # --- Size N=2 ---
    # ID 2: Simple return (Home ↔ Loc1) — most common motif
    library[index] = _DiGraph(n_nodes=2, edges=frozenset({(0, 1), (1, 0)}))
    index += 1

    # --- Size N=3 ---
    # ID 3
    library[index] = _DiGraph(
        n_nodes=3, edges=frozenset({(0, 1), (1, 0), (0, 2), (2, 0)})
    )
    index += 1

    # ID 4
    library[index] = _DiGraph(n_nodes=3, edges=frozenset({(0, 1), (1, 2), (2, 0)}))
    index += 1

    # ID 5
    library[index] = _DiGraph(
        n_nodes=3, edges=frozenset({(0, 1), (0, 2), (2, 1), (1, 0)})
    )
    index += 1

    # --- Size N=4 ---
    # ID 6
    library[index] = _DiGraph(
        n_nodes=4, edges=frozenset({(0, 1), (1, 3), (3, 0), (0, 2), (2, 0)})
    )
    index += 1

    # ID 7
    library[index] = _DiGraph(
        n_nodes=4, edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 0)})
    )
    index += 1

    # ID 8
    library[index] = _DiGraph(
        n_nodes=4,
        edges=frozenset({(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)}),
    )
    index += 1

    # ID 9
    library[index] = _DiGraph(
        n_nodes=4,
        edges=frozenset({(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)}),
    )
    index += 1

    # --- Size N=5 ---
    # ID 10
    library[index] = _DiGraph(
        n_nodes=5,
        edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 0)}),
    )
    index += 1

    # ID 11
    library[index] = _DiGraph(
        n_nodes=5, edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)})
    )
    index += 1

    # ID 12
    library[index] = _DiGraph(
        n_nodes=5,
        edges=frozenset({(0, 1), (1, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)}),
    )
    index += 1

    # ID 13
    library[index] = _DiGraph(
        n_nodes=5,
        edges=frozenset({(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)}),
    )
    index += 1

    # --- Size N=6 ---
    # ID 14
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (5, 0)}),
    )
    index += 1

    # ID 15
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)}),
    )
    index += 1

    # ID 16
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset(
            {(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 4), (4, 5), (5, 0)}
        ),
    )
    index += 1

    # ID 17
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset({(0, 1), (1, 2), (2, 0), (0, 3), (3, 5), (5, 4), (4, 0)}),
    )
    index += 1

    return library


def _motif_id(graph: _DiGraph) -> str:
    """Return the canonical motif ID string for *graph*.

    Format: ``m{n_nodes}:{canonical_bits}`` where ``canonical_bits`` is the
    zero-padded big-endian binary string produced by
    ``_canonical_adjacency_form``.

    Parameters
    ----------
    graph:
        The directed graph to identify.

    Returns
    -------
    str
        Canonical motif ID, e.g. ``"m1:0"`` for the single-node motif and
        ``"m2:0110"`` for the simple bidirectional return motif.

    Examples
    --------
    >>> g = _DiGraph(n_nodes=2, edges=frozenset({(0, 1), (1, 0)}))
    >>> _motif_id(g)
    'm2:0110'
    """
    if graph.n_nodes < 1:
        raise ValueError("Graph must have at least one node")
    if graph.n_nodes > 6:
        print(
            f"Warning: motif ID for graph with {graph.n_nodes} nodes may not be canonical (library only has motifs up to 6 nodes)"
        )
        return "-1"

    bits = _canonical_adjacency_form(graph.n_nodes, graph.edges)
    return f"m{graph.n_nodes}:{bits}"


# ---------------------------------------------------------------------------
# DataFrame → graph pipeline
# ---------------------------------------------------------------------------


def _compute_primary_home_node_id(nw_df: nw.DataFrame) -> str | None:
    """Identify the primary home node ID from a user's visit Narwhals DataFrame.

    Uses the ``"unique_id"`` and ``"purpose"`` columns. Picks the most
    night-time-visited home node (or, failing that, the most frequent home).

    Parameters
    ----------
    nw_df:
        A Narwhals DataFrame for a single user with columns ``"unique_id"``,
        ``"purpose"``, ``"start_timestamp"``, and optionally
        ``"duration_minutes"``.

    Returns
    -------
    str | None
        The primary home node identifier, or None if no HOME rows exist.
    """
    home_df = nw_df.filter(nw.col("purpose") == "HOME")
    if len(home_df) == 0:
        return None

    home_with_hour = home_df.with_columns(
        nw.col("start_timestamp").dt.hour().alias("start_hour")
    )
    night_df = home_with_hour.filter(
        (nw.col("start_hour") >= 22) | (nw.col("start_hour") < 6)
    )

    if len(night_df) > 0 and "duration_minutes" in night_df.columns:
        duration_by_home = (
            night_df.group_by("unique_id")
            .agg(nw.col("duration_minutes").sum())
            .sort("duration_minutes", descending=True)
        )
        return str(duration_by_home.row(0)[0])

    # Fallback: most frequent home node
    counts = (
        home_df.group_by("unique_id")
        .agg(nw.len().alias("_count"))
        .sort("_count", descending=True)
    )
    return str(counts.row(0)[0])


def build_motif_graph(
    daily_df: Any,
    primary_home_node_id: str,
    last_night_node_id: str | None = None,
    next_day_first_node_id: str | None = None,
    location_id_col: str = "unique_id",
    timestamp_col: str = "start_timestamp",
    home_suffix: str = "_HOME",
) -> _DiGraph:
    """Construct the directed mobility graph for a single user-day.

    Accepts any Narwhals-compatible DataFrame (or a Narwhals DataFrame).
    The graph is built from the sorted location sequence — no intermediate
    DataFrame construction is needed after extraction.

    Parameters
    ----------
    daily_df:
        DataFrame for the specific day. Must contain ``location_id_col``
        and ``timestamp_col`` columns.
    primary_home_node_id:
        Identifier for the primary home node (used as node 0 anchor and
        as loop-closure target when the day does not end at home).
    last_night_node_id:
        If provided, prepended as the day's starting location.
    next_day_first_node_id:
        If provided and ends with ``home_suffix``, used as the loop-closure
        target instead of ``primary_home_node_id``.
    location_id_col:
        Column name for the unique location identifier.  Default ``"unique_id"``.
    timestamp_col:
        Column name for the visit start time.  Default ``"start_timestamp"``.
    home_suffix:
        Suffix that identifies HOME nodes.  Default ``"_HOME"``.

    Returns
    -------
    _DiGraph
        The directed mobility graph for the day.
    """
    if primary_home_node_id is None:
        raise ValueError("primary_home_node_id cannot be None")

    # Accept either native or already-wrapped Narwhals DataFrames
    if not isinstance(daily_df, nw.DataFrame):
        nw_daily = nw.from_native(daily_df, eager_only=True)
    else:
        nw_daily = daily_df

    if len(nw_daily) == 0:
        return _DiGraph(n_nodes=1, edges=frozenset())

    # Sort by timestamp and extract the location sequence as a plain list.
    # No prefix-row DataFrame construction is needed: prepend as a Python string.
    sequence: list[str] = nw_daily.sort(timestamp_col)[location_id_col].to_list()

    # Prepend the starting node (last night's location or primary home)
    start_node = last_night_node_id if last_night_node_id else primary_home_node_id
    sequence = [start_node] + sequence

    # Loop closure: ensure day ends at a home node
    last_node = sequence[-1]
    if not last_node.endswith(home_suffix):
        if next_day_first_node_id and next_day_first_node_id.endswith(home_suffix):
            sequence.append(next_day_first_node_id)
        else:
            sequence.append(primary_home_node_id)

    # Remove consecutive duplicates (self-loops)
    cleaned: list[str] = [sequence[0]]
    for node in sequence[1:]:
        if node != cleaned[-1]:
            cleaned.append(node)

    # Map node names to integers; force primary_home to be node 0
    unique_nodes = sorted(set(cleaned))
    if primary_home_node_id in unique_nodes:
        unique_nodes.remove(primary_home_node_id)
        unique_nodes = [primary_home_node_id] + unique_nodes

    id_map = {name: i for i, name in enumerate(unique_nodes)}

    # Collect edges (unweighted, deduplicated)
    edges: set[tuple[int, int]] = set()
    for i in range(len(cleaned) - 1):
        u = id_map[cleaned[i]]
        v = id_map[cleaned[i + 1]]
        edges.add((u, v))

    return _DiGraph(n_nodes=len(unique_nodes), edges=frozenset(edges))


def _build_daily_motif_records(
    user_nw_df: nw.DataFrame,
    user_id_col: str,
    location_id_col: str,
) -> list[dict]:
    """Build per-day motif records for a single user.

    Parameters
    ----------
    user_nw_df:
        Narwhals DataFrame for one user.  Must contain ``location_id_col``,
        ``"purpose"``, ``"start_timestamp"``, ``"end_timestamp"``.
    user_id_col:
        Column name for the user identifier.
    location_id_col:
        Column name for the raw location identifier (before compositing with purpose).

    Returns
    -------
    list[dict]
        One dict per day containing ``user_id_col``, ``"date"``,
        ``"num_nodes"``, ``"num_edges"``, and ``"graph"`` keys.
    """
    df = (
        user_nw_df.sort("start_timestamp")
        .with_columns(
            nw.col("start_timestamp").cast(nw.Datetime).alias("start_timestamp"),
            nw.col("end_timestamp").cast(nw.Datetime).alias("end_timestamp"),
            (
                nw.col(location_id_col).cast(nw.String)
                + nw.lit("_")
                + nw.col("purpose").cast(nw.String)
            ).alias("unique_id"),
        )
        .with_columns(
            # Use truncate to day-boundary rather than .dt.date() — the latter
            # raises NotImplementedError on the default pandas backend because
            # pandas .dt.date returns an object-dtype Series.
            nw.col("start_timestamp")
            .dt.truncate("1d")
            .alias("date"),
        )
    )

    primary_home_node_id = _compute_primary_home_node_id(df)
    if primary_home_node_id is None:
        return []

    # Extract unique dates in sorted order as Python date objects
    unique_dates = sorted(set(df["date"].to_list()))

    records = []
    for i, current_date in enumerate(unique_dates):
        daily_df = df.filter(nw.col("date") == current_date).sort("start_timestamp")
        if len(daily_df) == 0:
            continue

        # Look ahead: first node of next day (if HOME before 03:00)
        next_day_first_node_id = None
        if i + 1 < len(unique_dates):
            next_day_df = (
                df.filter(nw.col("date") == unique_dates[i + 1])
                .sort("start_timestamp")
                .with_columns(nw.col("start_timestamp").dt.hour().alias("_hour"))
            )
            if len(next_day_df) > 0:
                first_uid = next_day_df["unique_id"].to_list()[0]
                first_hour = next_day_df["_hour"].to_list()[0]
                if first_hour < 3 and first_uid.endswith("_HOME"):
                    next_day_first_node_id = first_uid

        # Look back: last node of previous day (if HOME and ended after 03:00)
        last_night_node_id = None
        if i - 1 >= 0:
            prev_day_df = (
                df.filter(nw.col("date") == unique_dates[i - 1])
                .sort("start_timestamp")
                .with_columns(nw.col("end_timestamp").dt.hour().alias("_end_hour"))
            )
            if len(prev_day_df) > 0:
                uids = prev_day_df["unique_id"].to_list()
                end_hours = prev_day_df["_end_hour"].to_list()
                last_uid = uids[-1]
                end_hour = end_hours[-1]
                if last_uid == primary_home_node_id and (
                    end_hour > 3 or end_hour == 23
                ):
                    last_night_node_id = last_uid

        graph = build_motif_graph(
            daily_df,
            primary_home_node_id=primary_home_node_id,
            last_night_node_id=last_night_node_id,
            next_day_first_node_id=next_day_first_node_id,
        )

        first_user_id = daily_df[user_id_col].to_list()[0]
        records.append(
            {
                user_id_col: first_user_id,
                "date": current_date,
                "num_nodes": graph.n_nodes,
                "num_edges": len(graph.edges),
                "motif_id": _motif_id(graph),
            }
        )

    return records


def discover_daily_motifs_from_agents(
    df: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    purpose_col: str | None = None,
    timestamp_col: str | None = None,
    end_timestamp_col: str | None = None,
    duration_col: str | None = None,
    show_progress: bool = False,
) -> tuple[Any, Any]:
    """Discover daily mobility motifs for all agents in a dataset.

    Parameters
    ----------
    df:
        Input visits DataFrame (any backend).  Must contain at minimum
        user_id, location_id, purpose, start_timestamp, and end_timestamp
        columns (or their auto-detected equivalents).
    user_id_col:
        User identifier column.  Auto-detected if None.
    location_id_col:
        Location identifier column.  Auto-detected if None.
    purpose_col:
        Activity purpose column (``"HOME"``, ``"WORK"``, etc.).
        Defaults to ``"purpose"`` if found.
    timestamp_col:
        Start-time column.  Defaults to ``"start_timestamp"`` if found.
    end_timestamp_col:
        End-time column.  Defaults to ``"end_timestamp"`` if found.
    duration_col:
        Duration column used for primary-home selection.  Defaults to
        ``"duration_minutes"`` if found.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        ``(daily_motifs_df, motif_distribution_df)`` in the same backend as
        the input ``df``.

        *daily_motifs_df*: one row per user-day with columns
        ``[user_id_col, "date", "motif_id", "num_nodes", "num_edges"]``.

        *motif_distribution_df*: one row per distinct motif with columns
        ``["motif_id", "count", "percentage"]``.
    """
    if show_progress:
        from tqdm import tqdm  # Lazy import to avoid dependency if not showing progress

    # End-timestamp candidates — not in _common.py (motifs-specific)
    _ETS_CANDIDATES = ["end_timestamp", "end_time"]

    nw_df = nw.from_native(df, eager_only=True)
    backend = nw_df.implementation
    cols = nw_df.columns

    # Column auto-detection using shared candidate lists from _common.py
    if user_id_col is None:
        user_id_col = _pick_existing_column(cols, USER_ID_CANDIDATES) or "agent_id"
    if location_id_col is None:
        location_id_col = (
            _pick_existing_column(cols, LOCATION_CANDIDATES) or "location_id"
        )
    if purpose_col is None:
        purpose_col = _pick_existing_column(cols, PURPOSE_CANDIDATES) or "purpose"
    if timestamp_col is None:
        timestamp_col = (
            _pick_existing_column(cols, TIMESTAMP_CANDIDATES) or "start_timestamp"
        )
    if end_timestamp_col is None:
        end_timestamp_col = (
            _pick_existing_column(cols, _ETS_CANDIDATES) or "end_timestamp"
        )
    if duration_col is None:
        duration_col = _pick_existing_column(cols, DURATION_CANDIDATES)

    # Rename columns to canonical internal names expected by helpers
    rename_map: dict[str, str] = {}
    if purpose_col != "purpose":
        rename_map[purpose_col] = "purpose"
    if timestamp_col != "start_timestamp":
        rename_map[timestamp_col] = "start_timestamp"
    if end_timestamp_col != "end_timestamp":
        rename_map[end_timestamp_col] = "end_timestamp"
    if duration_col and duration_col != "duration_minutes":
        rename_map[duration_col] = "duration_minutes"

    work_df = nw_df.rename(rename_map).sort([user_id_col, "start_timestamp"])

    # Build per-day records for each user by splitting on user_id_col
    user_ids = sorted(set(work_df[user_id_col].to_list()))
    all_records: list[dict] = []
    user_iterator = (
        user_ids if not show_progress else tqdm(user_ids, desc="Processing users")
    )
    for uid in user_iterator:
        user_group = work_df.filter(nw.col(user_id_col) == uid)
        records = _build_daily_motif_records(
            user_group, user_id_col=user_id_col, location_id_col=location_id_col
        )
        all_records.extend(records)

    if not all_records:
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

    # Assemble daily_motifs_df from records (drop the graph object)
    daily_data: dict[str, list] = {
        user_id_col: [r[user_id_col] for r in all_records],
        "date": [r["date"] for r in all_records],
        "num_nodes": [r["num_nodes"] for r in all_records],
        "num_edges": [r["num_edges"] for r in all_records],
        "motif_id": [r["motif_id"] for r in all_records],
    }
    daily_nw = nw.from_dict(daily_data, backend=backend).sort(
        ["motif_id", user_id_col, "date"]
    )

    return daily_nw.to_native()


def compute_daily_motifs_distribution(
    daily_motifs_df: Any,
    motif_id_col: str = "motif_id",
) -> Any:
    """Compute the distribution of motifs from a daily motifs DataFrame.

    Parameters
    ----------
    daily_motifs_df:
        DataFrame containing at least a column with motif IDs (e.g. output
        from discover_daily_motifs_from_agents).
    motif_id_col:
        Column name for the motif ID.  Default ``"motif_id"``.
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
