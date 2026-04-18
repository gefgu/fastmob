"""Mobility motif classification without networkx.

Implements the directed-graph motif library from Pappalardo et al. using
plain Python adjacency-list representations for full independence from
the networkx library.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd

from ._common import _pick_existing_column, USER_ID_CANDIDATES, LOCATION_CANDIDATES


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


def _is_isomorphic(g1: _DiGraph, g2: _DiGraph) -> bool:
    """Brute-force isomorphism check for directed graphs.

    Works for up to ~8 nodes (6! = 720 permutations; 8! = 40 320 — fast
    enough for the static motif library which has at most 6 nodes).

    Parameters
    ----------
    g1, g2:
        The two directed graphs to compare.

    Returns
    -------
    bool
        True if g1 and g2 are isomorphic (there exists a bijection between
        their node sets that preserves all directed edges).
    """
    if g1.n_nodes != g2.n_nodes or len(g1.edges) != len(g2.edges):
        return False

    nodes = list(range(g1.n_nodes))
    for perm in permutations(nodes):
        mapping = {i: perm[i] for i in nodes}
        mapped_edges = frozenset((mapping[u], mapping[v]) for u, v in g1.edges)
        if mapped_edges == g2.edges:
            return True

    return False


# ---------------------------------------------------------------------------
# Static motif library (17 canonical mobility patterns)
# ---------------------------------------------------------------------------

def get_motif_library() -> dict[int, _DiGraph]:
    """Return the 17 canonical directed mobility motifs.

    The motifs replicate the networkx-based library from the source
    ``mobility_analysis/measures/motifs.py``, translated to ``_DiGraph``.

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
    library[index] = _DiGraph(
        n_nodes=3, edges=frozenset({(0, 1), (1, 2), (2, 0)})
    )
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
        edges=frozenset(
            {(0, 1), (1, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)}
        ),
    )
    index += 1

    # ID 13
    library[index] = _DiGraph(
        n_nodes=5,
        edges=frozenset(
            {(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)}
        ),
    )
    index += 1

    # --- Size N=6 ---
    # ID 14
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset(
            {(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (5, 0)}
        ),
    )
    index += 1

    # ID 15
    library[index] = _DiGraph(
        n_nodes=6,
        edges=frozenset(
            {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)}
        ),
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
        edges=frozenset(
            {(0, 1), (1, 2), (2, 0), (0, 3), (3, 5), (5, 4), (4, 0)}
        ),
    )
    index += 1

    return library


# ---------------------------------------------------------------------------
# Dynamic library management
# ---------------------------------------------------------------------------

def format_motif_id_v2(num_nodes: int, num_edges: int, order: int) -> str:
    """Format a motif ID string as ``{n_nodes}_{n_edges}_{order}``."""
    return f"{num_nodes}_{num_edges}_{order}"


def create_dynamic_motif_library_object_from_static() -> dict[int, dict[int, list[_DiGraph]]]:
    """Create a dynamic motif library initialized from the static 17-motif library.

    Returns
    -------
    dict[int, dict[int, list[_DiGraph]]]
        Structure: ``{n_nodes: {n_edges: [_DiGraph, ...]}}``.
    """
    static = get_motif_library()
    dynamic: dict[int, dict[int, list[_DiGraph]]] = {}

    for _, graph in static.items():
        n_nodes = graph.n_nodes
        n_edges = len(graph.edges)
        dynamic.setdefault(n_nodes, {}).setdefault(n_edges, []).append(graph)

    return dynamic


def save_dynamic_motif_library(dynamic_library: dict, file_path: str | Path) -> None:
    """Pickle the dynamic library to *file_path*.

    Parameters
    ----------
    dynamic_library:
        The dynamic motif library to save.
    file_path:
        Destination file path (must be provided — no default).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dynamic_library, fh)


def load_dynamic_motif_library(
    file_path: str | Path,
    initialize_if_missing: bool = True,
) -> dict:
    """Load the dynamic motif library from *file_path*.

    Parameters
    ----------
    file_path:
        Path to the pickle file (must be provided — no default).
    initialize_if_missing:
        When True (default), create and save a new library from the static
        motifs if the file does not exist.  When False, raise
        ``FileNotFoundError``.

    Returns
    -------
    dict
        The dynamic motif library.
    """
    path = Path(file_path)
    if not path.exists():
        if not initialize_if_missing:
            raise FileNotFoundError(f"Dynamic motif library not found: {path}")
        dynamic = create_dynamic_motif_library_object_from_static()
        save_dynamic_motif_library(dynamic, path)
        return dynamic

    with open(path, "rb") as fh:
        return pickle.load(fh)


def classify_or_add_motif_v2(
    user_graph: _DiGraph,
    dynamic_library: dict,
    auto_add_new_motifs: bool = True,
) -> tuple[str | None, bool]:
    """Classify *user_graph* against *dynamic_library* using isomorphism.

    Parameters
    ----------
    user_graph:
        The mobility graph for a user-day.
    dynamic_library:
        Mutable nested dict ``{n_nodes: {n_edges: [_DiGraph, ...]}}``.
        Modified in-place when a new motif is added.
    auto_add_new_motifs:
        When True (default), novel graphs are added to the library and
        assigned a new ID.  When False, novel graphs return ``(None, False)``.

    Returns
    -------
    tuple[str | None, bool]
        ``(motif_id, is_newly_added)``.  ``motif_id`` follows the
        ``"{n_nodes}_{n_edges}_{order}"`` format.
    """
    n_nodes = user_graph.n_nodes
    n_edges = len(user_graph.edges)

    buckets_by_edges = dynamic_library.setdefault(n_nodes, {})
    graph_bucket: list[_DiGraph] = buckets_by_edges.setdefault(n_edges, [])

    for idx, reference_graph in enumerate(graph_bucket, start=1):
        if _is_isomorphic(user_graph, reference_graph):
            return format_motif_id_v2(n_nodes, n_edges, idx), False

    if not auto_add_new_motifs:
        return None, False

    graph_bucket.append(user_graph)
    new_id = format_motif_id_v2(n_nodes, n_edges, len(graph_bucket))
    return new_id, True


# ---------------------------------------------------------------------------
# DataFrame → graph pipeline
# ---------------------------------------------------------------------------

def _compute_primary_home_node_id(df: pd.DataFrame) -> str | None:
    """Identify the primary home node ID from a user's visit DataFrame.

    Uses the ``"unique_id"`` and ``"purpose"`` columns. Picks the most
    night-time-visited home node (or, failing that, the most frequent home).
    """
    home_rows = df[df["purpose"] == "HOME"]
    if home_rows.empty:
        return None

    home_with_times = home_rows.copy()
    home_with_times["start_hour"] = home_with_times["start_timestamp"].dt.hour
    night_mask = (home_with_times["start_hour"] >= 22) | (
        home_with_times["start_hour"] < 6
    )
    night_homes = home_with_times[night_mask].copy()

    if not night_homes.empty and "duration_minutes" in night_homes.columns:
        duration_by_home = night_homes.groupby("unique_id")["duration_minutes"].sum()
        return duration_by_home.idxmax()

    return home_rows["unique_id"].value_counts().idxmax()


def build_motif_graph(
    daily_df: pd.DataFrame,
    primary_home_node_id: str,
    last_night_node_id: str | None = None,
    next_day_first_node_id: str | None = None,
    location_id_col: str = "unique_id",
    timestamp_col: str = "start_timestamp",
    home_suffix: str = "_HOME",
) -> _DiGraph:
    """Construct the directed mobility graph for a single user-day.

    Parameters
    ----------
    daily_df:
        DataFrame for the specific day. Must contain ``location_id_col``
        and ``timestamp_col`` columns.
    primary_home_node_id:
        Identifier for the primary home node (used as node 0 anchor and
        as loop-closure target when the day does not end at home).
    last_night_node_id:
        If provided, the day is prepended with this node at ``timestamp - 1s``.
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

    if daily_df.empty:
        return _DiGraph(n_nodes=1, edges=frozenset())

    df = daily_df.copy()

    # Prepend last-night or primary-home as day's starting node
    if last_night_node_id:
        prefix_row = pd.DataFrame({
            location_id_col: [last_night_node_id],
            timestamp_col: [df[timestamp_col].min() - pd.Timedelta(seconds=1)],
        })
    else:
        prefix_row = pd.DataFrame({
            location_id_col: [primary_home_node_id],
            timestamp_col: [df[timestamp_col].min() - pd.Timedelta(seconds=1)],
        })

    df = (
        pd.concat([prefix_row, df], ignore_index=True)
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    sequence = df[location_id_col].tolist()

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
    user_df: pd.DataFrame,
    user_id_col: str,
    location_id_col: str,
) -> list[dict]:
    """Build per-day motif records for a single user.

    Parameters
    ----------
    user_df:
        Visits DataFrame for one user.  Must contain ``location_id_col``,
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
    df = user_df.sort_values("start_timestamp").reset_index(drop=True).copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])
    df["date"] = df["start_timestamp"].dt.date
    df["unique_id"] = df[location_id_col].astype(str) + "_" + df["purpose"].astype(str)

    primary_home_node_id = _compute_primary_home_node_id(df)
    if primary_home_node_id is None:
        return []

    records = []
    unique_dates = sorted(df["date"].unique())

    for i, current_date in enumerate(unique_dates):
        daily_df = (
            df[df["date"] == current_date]
            .reset_index(drop=True)
            .sort_values("start_timestamp")
        )
        if daily_df.empty:
            continue

        # Look ahead: first node of next day (if HOME before 03:00)
        next_day_first_node_id = None
        if i + 1 < len(unique_dates):
            next_day_df = df[df["date"] == unique_dates[i + 1]].sort_values("start_timestamp")
            if not next_day_df.empty:
                first_row = next_day_df.iloc[0]
                if (first_row["start_timestamp"].hour < 3
                        and first_row["unique_id"].endswith("_HOME")):
                    next_day_first_node_id = first_row["unique_id"]

        # Look back: last node of previous day (if HOME and ended after 03:00 / before midnight)
        last_night_node_id = None
        if i - 1 >= 0:
            prev_day_df = df[df["date"] == unique_dates[i - 1]].sort_values("start_timestamp")
            if not prev_day_df.empty:
                last_loc = prev_day_df.iloc[-1]["unique_id"]
                end_hour = prev_day_df["end_timestamp"].iloc[-1].hour
                if last_loc == primary_home_node_id and (end_hour > 3 or end_hour == 23):
                    last_night_node_id = last_loc

        graph = build_motif_graph(
            daily_df,
            primary_home_node_id=primary_home_node_id,
            last_night_node_id=last_night_node_id,
            next_day_first_node_id=next_day_first_node_id,
        )

        records.append({
            user_id_col: daily_df.iloc[0][user_id_col],
            "date": current_date,
            "num_nodes": graph.n_nodes,
            "num_edges": len(graph.edges),
            "graph": graph,
        })

    return records


def discover_daily_motifs_from_agents(
    df: Any,
    user_id_col: str | None = None,
    location_id_col: str | None = None,
    purpose_col: str | None = None,
    timestamp_col: str | None = None,
    end_timestamp_col: str | None = None,
    duration_col: str | None = None,
    dynamic_library_file: str | None = None,
    auto_add_new_motifs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    dynamic_library_file:
        Path to the persisted dynamic library pickle file.  When None,
        an in-memory library is used (no disk I/O).  When provided, the
        library is loaded from this path (and saved if new motifs are added).
    auto_add_new_motifs:
        When True (default), novel graphs are added to the dynamic library.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(daily_motifs_df, motif_distribution_df)``.

        *daily_motifs_df*: one row per user-day with columns
        ``[user_id_col, "date", "motif_id", "num_nodes", "num_edges"]``.

        *motif_distribution_df*: one row per distinct motif with columns
        ``["motif_id", "count", "percentage"]``.
    """
    import narwhals as nw

    nw_df = nw.from_native(df, eager_only=True)
    cols = nw_df.columns

    # Column auto-detection
    _UID_CANDIDATES = ["user_id", "uid", "agent_id", "user"]
    _LOC_CANDIDATES = ["location_id", "area", "venueId"]
    _PURPOSE_CANDIDATES = ["purpose", "activity", "location_type"]
    _TS_CANDIDATES = ["start_timestamp", "timestamp", "datetime"]
    _ETS_CANDIDATES = ["end_timestamp", "end_time"]
    _DUR_CANDIDATES = ["duration_minutes", "duration"]

    if user_id_col is None:
        user_id_col = _pick_existing_column(cols, _UID_CANDIDATES) or "agent_id"
    if location_id_col is None:
        location_id_col = _pick_existing_column(cols, _LOC_CANDIDATES) or "location_id"
    if purpose_col is None:
        purpose_col = _pick_existing_column(cols, _PURPOSE_CANDIDATES) or "purpose"
    if timestamp_col is None:
        timestamp_col = _pick_existing_column(cols, _TS_CANDIDATES) or "start_timestamp"
    if end_timestamp_col is None:
        end_timestamp_col = _pick_existing_column(cols, _ETS_CANDIDATES) or "end_timestamp"
    if duration_col is None:
        duration_col = _pick_existing_column(cols, _DUR_CANDIDATES)

    # Normalize to pandas
    native = nw_df.to_native()
    if not isinstance(native, pd.DataFrame):
        native = pd.DataFrame(native)

    # Rename columns to expected names for internal helpers
    rename_map: dict[str, str] = {}
    if purpose_col != "purpose":
        rename_map[purpose_col] = "purpose"
    if timestamp_col != "start_timestamp":
        rename_map[timestamp_col] = "start_timestamp"
    if end_timestamp_col != "end_timestamp":
        rename_map[end_timestamp_col] = "end_timestamp"
    if duration_col and duration_col != "duration_minutes":
        rename_map[duration_col] = "duration_minutes"

    work_df = native.rename(columns=rename_map).copy()

    # Sort globally
    work_df = work_df.sort_values([user_id_col, "start_timestamp"]).reset_index(drop=True)

    # Load (or create) the dynamic library
    dynamic_library_changed = False
    if dynamic_library_file is not None:
        dynamic_library = load_dynamic_motif_library(dynamic_library_file)
    else:
        dynamic_library = create_dynamic_motif_library_object_from_static()

    # Build per-day records for each user
    all_records: list[dict] = []
    for _, user_group in work_df.groupby(user_id_col, sort=True):
        records = _build_daily_motif_records(
            user_group, user_id_col=user_id_col, location_id_col=location_id_col
        )
        all_records.extend(records)

    if not all_records:
        empty_daily = pd.DataFrame(
            columns=[user_id_col, "date", "motif_id", "num_nodes", "num_edges"]
        )
        empty_dist = pd.DataFrame(columns=["motif_id", "count", "percentage"])
        return empty_daily, empty_dist

    # Classify each daily graph
    assigned_motif_ids: list[str] = []
    for record in all_records:
        graph = record["graph"]
        motif_id, is_new = classify_or_add_motif_v2(
            graph, dynamic_library=dynamic_library, auto_add_new_motifs=auto_add_new_motifs
        )
        if motif_id is None:
            motif_id = "-1"
        if is_new:
            dynamic_library_changed = True
        assigned_motif_ids.append(motif_id)

    # Persist library if it changed and a file path was given
    if dynamic_library_file is not None and dynamic_library_changed:
        save_dynamic_motif_library(dynamic_library, dynamic_library_file)

    # Assemble result DataFrames
    daily_motifs_df = pd.DataFrame(all_records)
    daily_motifs_df["motif_id"] = assigned_motif_ids
    daily_motifs_df = (
        daily_motifs_df.drop(columns=["graph"])
        .sort_values(["motif_id", user_id_col, "date"])
        .reset_index(drop=True)
    )

    motif_dist_df = (
        daily_motifs_df["motif_id"]
        .value_counts()
        .sort_index()
        .rename_axis("motif_id")
        .reset_index(name="count")
    )
    motif_dist_df["percentage"] = (
        motif_dist_df["count"] / motif_dist_df["count"].sum()
    ) * 100

    return daily_motifs_df, motif_dist_df
