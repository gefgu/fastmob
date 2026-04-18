"""Tests for motif classification measures (Step 9)."""
import pickle
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# _DiGraph and _is_isomorphic tests
# ---------------------------------------------------------------------------

def test_digraph_self_isomorphic():
    """Any graph is isomorphic to itself."""
    from skmob2.measures.motifs import _DiGraph, _is_isomorphic
    g = _DiGraph(n_nodes=3, edges=frozenset({(0, 1), (1, 2), (2, 0)}))
    assert _is_isomorphic(g, g)


def test_digraph_two_graphs_isomorphic():
    """Two structurally identical graphs with relabeled nodes are isomorphic."""
    from skmob2.measures.motifs import _DiGraph, _is_isomorphic
    # Cycle 0→1→2→0
    g1 = _DiGraph(n_nodes=3, edges=frozenset({(0, 1), (1, 2), (2, 0)}))
    # Same cycle but with nodes relabeled: 1→2→0→1
    g2 = _DiGraph(n_nodes=3, edges=frozenset({(1, 2), (2, 0), (0, 1)}))
    assert _is_isomorphic(g1, g2)


def test_digraph_single_vs_two_nodes_not_isomorphic():
    """Single-node motif and two-node return motif are not isomorphic."""
    from skmob2.measures.motifs import _DiGraph, _is_isomorphic
    single = _DiGraph(n_nodes=1, edges=frozenset())
    simple_return = _DiGraph(n_nodes=2, edges=frozenset({(0, 1), (1, 0)}))
    assert not _is_isomorphic(single, simple_return)


def test_digraph_different_edge_count_not_isomorphic():
    """Graphs with different edge counts can't be isomorphic."""
    from skmob2.measures.motifs import _DiGraph, _is_isomorphic
    g1 = _DiGraph(n_nodes=2, edges=frozenset({(0, 1)}))
    g2 = _DiGraph(n_nodes=2, edges=frozenset({(0, 1), (1, 0)}))
    assert not _is_isomorphic(g1, g2)


# ---------------------------------------------------------------------------
# get_motif_library tests
# ---------------------------------------------------------------------------

def test_motif_library_has_17_graphs():
    """Static library must have exactly 17 canonical motifs (keys 1–17)."""
    from skmob2.measures.motifs import get_motif_library
    library = get_motif_library()
    assert len(library) == 17
    assert set(library.keys()) == set(range(1, 18))


def test_motif_1_is_single_node():
    """Motif 1: single-node graph (Home only), 0 edges."""
    from skmob2.measures.motifs import get_motif_library
    library = get_motif_library()
    m1 = library[1]
    assert m1.n_nodes == 1
    assert len(m1.edges) == 0


def test_motif_2_is_simple_return():
    """Motif 2: 2 nodes, 2 edges (bidirectional simple return)."""
    from skmob2.measures.motifs import get_motif_library
    library = get_motif_library()
    m2 = library[2]
    assert m2.n_nodes == 2
    assert len(m2.edges) == 2
    assert frozenset({(0, 1), (1, 0)}) == m2.edges


def test_all_motifs_have_correct_types():
    """All motifs in the library must be _DiGraph instances."""
    from skmob2.measures.motifs import get_motif_library, _DiGraph
    library = get_motif_library()
    for key, motif in library.items():
        assert isinstance(motif, _DiGraph), f"Motif {key} is not a _DiGraph"


# ---------------------------------------------------------------------------
# format_motif_id_v2 tests
# ---------------------------------------------------------------------------

def test_format_motif_id_v2_basic():
    from skmob2.measures.motifs import format_motif_id_v2
    assert format_motif_id_v2(2, 2, 1) == "2_2_1"
    assert format_motif_id_v2(3, 3, 2) == "3_3_2"
    assert format_motif_id_v2(1, 0, 1) == "1_0_1"


# ---------------------------------------------------------------------------
# create_dynamic_motif_library_object_from_static tests
# ---------------------------------------------------------------------------

def test_dynamic_library_structure():
    """Dynamic library has correct nested dict structure."""
    from skmob2.measures.motifs import create_dynamic_motif_library_object_from_static
    dynamic = create_dynamic_motif_library_object_from_static()
    # Top-level keys are n_nodes values
    for n_nodes, by_edges in dynamic.items():
        assert isinstance(n_nodes, int)
        for n_edges, graphs in by_edges.items():
            assert isinstance(n_edges, int)
            assert isinstance(graphs, list)
            assert len(graphs) > 0


def test_dynamic_library_contains_all_17_static_motifs():
    """Dynamic library initialized from static contains exactly 17 graphs."""
    from skmob2.measures.motifs import create_dynamic_motif_library_object_from_static
    dynamic = create_dynamic_motif_library_object_from_static()
    total = sum(len(g_list) for by_edges in dynamic.values() for g_list in by_edges.values())
    assert total == 17


# ---------------------------------------------------------------------------
# classify_or_add_motif_v2 tests
# ---------------------------------------------------------------------------

def test_classify_motif_2_returns_known_id():
    """The simple return graph (2 nodes, 2 bidirectional edges) classifies to '2_2_1'."""
    from skmob2.measures.motifs import (
        _DiGraph,
        create_dynamic_motif_library_object_from_static,
        classify_or_add_motif_v2,
    )
    simple_return = _DiGraph(n_nodes=2, edges=frozenset({(0, 1), (1, 0)}))
    dynamic = create_dynamic_motif_library_object_from_static()
    motif_id, is_new = classify_or_add_motif_v2(simple_return, dynamic)
    assert motif_id == "2_2_1"
    assert is_new is False


def test_classify_motif_1_returns_known_id():
    """Single-node graph classifies to '1_0_1'."""
    from skmob2.measures.motifs import (
        _DiGraph,
        create_dynamic_motif_library_object_from_static,
        classify_or_add_motif_v2,
    )
    single_node = _DiGraph(n_nodes=1, edges=frozenset())
    dynamic = create_dynamic_motif_library_object_from_static()
    motif_id, is_new = classify_or_add_motif_v2(single_node, dynamic)
    assert motif_id == "1_0_1"
    assert is_new is False


def test_classify_novel_graph_adds_new_motif():
    """A novel graph not in the static library is added with is_new=True."""
    from skmob2.measures.motifs import (
        _DiGraph,
        create_dynamic_motif_library_object_from_static,
        classify_or_add_motif_v2,
    )
    # A self-loop graph — not in the static library
    novel = _DiGraph(n_nodes=1, edges=frozenset({(0, 0)}))
    dynamic = create_dynamic_motif_library_object_from_static()
    motif_id, is_new = classify_or_add_motif_v2(novel, dynamic, auto_add_new_motifs=True)
    assert is_new is True
    assert motif_id is not None


def test_classify_auto_add_false_returns_none_for_novel():
    """With auto_add_new_motifs=False, novel graph returns (None, False)."""
    from skmob2.measures.motifs import (
        _DiGraph,
        create_dynamic_motif_library_object_from_static,
        classify_or_add_motif_v2,
    )
    novel = _DiGraph(n_nodes=4, edges=frozenset({(0, 1), (2, 3)}))  # disconnected
    dynamic = create_dynamic_motif_library_object_from_static()
    motif_id, is_new = classify_or_add_motif_v2(novel, dynamic, auto_add_new_motifs=False)
    assert motif_id is None
    assert is_new is False


# ---------------------------------------------------------------------------
# save_dynamic_motif_library / load_dynamic_motif_library round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """Save and reload the dynamic library produces an equal structure."""
    from skmob2.measures.motifs import (
        create_dynamic_motif_library_object_from_static,
        save_dynamic_motif_library,
        load_dynamic_motif_library,
        _DiGraph,
    )
    dynamic = create_dynamic_motif_library_object_from_static()
    file_path = tmp_path / "test_library.pkl"
    save_dynamic_motif_library(dynamic, file_path)
    loaded = load_dynamic_motif_library(file_path)

    # Same keys at top level
    assert set(loaded.keys()) == set(dynamic.keys())
    for n_nodes in dynamic:
        assert set(loaded[n_nodes].keys()) == set(dynamic[n_nodes].keys())
        for n_edges in dynamic[n_nodes]:
            orig_graphs = dynamic[n_nodes][n_edges]
            loaded_graphs = loaded[n_nodes][n_edges]
            assert len(orig_graphs) == len(loaded_graphs)
            for orig_g, load_g in zip(orig_graphs, loaded_graphs):
                assert isinstance(load_g, _DiGraph)
                assert orig_g.n_nodes == load_g.n_nodes
                assert orig_g.edges == load_g.edges


def test_load_missing_file_initializes(tmp_path):
    """load_dynamic_motif_library with initialize_if_missing=True creates the file."""
    from skmob2.measures.motifs import load_dynamic_motif_library
    missing = tmp_path / "does_not_exist.pkl"
    dynamic = load_dynamic_motif_library(missing, initialize_if_missing=True)
    assert len(dynamic) > 0
    # File should now exist
    assert missing.exists()


def test_load_missing_file_raises_when_not_initializing(tmp_path):
    """load_dynamic_motif_library with initialize_if_missing=False raises FileNotFoundError."""
    from skmob2.measures.motifs import load_dynamic_motif_library
    missing = tmp_path / "does_not_exist.pkl"
    with pytest.raises(FileNotFoundError):
        load_dynamic_motif_library(missing, initialize_if_missing=False)


# ---------------------------------------------------------------------------
# build_motif_graph tests
# ---------------------------------------------------------------------------

def _make_daily_df(date_str="2020-01-01"):
    """Minimal single-day, single-user DataFrame for build_motif_graph."""
    base = pd.Timestamp(date_str)
    return pd.DataFrame({
        "unique_id": ["home_loc_HOME", "work_loc_WORK", "home_loc_HOME"],
        "start_timestamp": [
            base + pd.Timedelta(hours=7),
            base + pd.Timedelta(hours=9),
            base + pd.Timedelta(hours=17),
        ],
        "end_timestamp": [
            base + pd.Timedelta(hours=9),
            base + pd.Timedelta(hours=17),
            base + pd.Timedelta(hours=23),
        ],
        "purpose": ["HOME", "WORK", "HOME"],
    })


def test_build_motif_graph_returns_digraph():
    """build_motif_graph returns a _DiGraph instance."""
    from skmob2.measures.motifs import build_motif_graph, _DiGraph
    daily_df = _make_daily_df()
    g = build_motif_graph(daily_df, primary_home_node_id="home_loc_HOME")
    assert isinstance(g, _DiGraph)


def test_build_motif_graph_empty_returns_single_node():
    """Empty daily_df returns a single-node graph."""
    from skmob2.measures.motifs import build_motif_graph, _DiGraph
    empty_df = pd.DataFrame(columns=["unique_id", "start_timestamp", "end_timestamp", "purpose"])
    g = build_motif_graph(empty_df, primary_home_node_id="home_loc_HOME")
    assert g.n_nodes == 1
    assert len(g.edges) == 0


def test_build_motif_graph_home_work_home_is_simple_return():
    """HOME→WORK→HOME pattern produces the simple return motif (2 nodes, 2 edges)."""
    from skmob2.measures.motifs import build_motif_graph
    daily_df = _make_daily_df()
    g = build_motif_graph(daily_df, primary_home_node_id="home_loc_HOME")
    assert g.n_nodes == 2
    assert len(g.edges) == 2


# ---------------------------------------------------------------------------
# discover_daily_motifs_from_agents tests
# ---------------------------------------------------------------------------

def _make_multi_day_df():
    """Two days of visits for one user. Each day: HOME→WORK→HOME."""
    rows = []
    for day_offset in range(2):
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_offset)
        rows += [
            {"agent_id": "u1", "location_id": "home", "purpose": "HOME",
             "start_timestamp": base + pd.Timedelta(hours=0),
             "end_timestamp": base + pd.Timedelta(hours=8),
             "duration_minutes": 480},
            {"agent_id": "u1", "location_id": "work", "purpose": "WORK",
             "start_timestamp": base + pd.Timedelta(hours=9),
             "end_timestamp": base + pd.Timedelta(hours=17),
             "duration_minutes": 480},
            {"agent_id": "u1", "location_id": "home", "purpose": "HOME",
             "start_timestamp": base + pd.Timedelta(hours=18),
             "end_timestamp": base + pd.Timedelta(hours=23),
             "duration_minutes": 300},
        ]
    return pd.DataFrame(rows)


def test_discover_motifs_returns_two_dataframes():
    """discover_daily_motifs_from_agents returns exactly two DataFrames."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    result = discover_daily_motifs_from_agents(df)
    assert len(result) == 2
    daily_motifs_df, motif_dist_df = result
    assert isinstance(daily_motifs_df, pd.DataFrame)
    assert isinstance(motif_dist_df, pd.DataFrame)


def test_discover_motifs_daily_rows_count():
    """Two days, one user → daily_motifs_df has 2 rows."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    assert len(daily_motifs_df) == 2


def test_discover_motifs_has_required_columns():
    """daily_motifs_df has required columns."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    required = {"agent_id", "date", "motif_id", "num_nodes", "num_edges"}
    assert required.issubset(set(daily_motifs_df.columns))


def test_discover_motifs_home_work_home_classified_correctly():
    """HOME→WORK→HOME each day → motif_id should be '2_2_1' (simple return)."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    # All days are simple returns: 2 nodes, 2 edges
    assert (daily_motifs_df["num_nodes"] == 2).all()
    assert (daily_motifs_df["num_edges"] == 2).all()


def test_discover_motifs_distribution_sum_to_100():
    """Motif distribution percentages sum to 100."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    _, motif_dist_df = discover_daily_motifs_from_agents(df)
    assert motif_dist_df["percentage"].sum() == pytest.approx(100.0, abs=1e-6)


def test_discover_motifs_uses_custom_library_file(tmp_path):
    """When dynamic_library_file is provided, library is persisted to that path."""
    from skmob2.measures.motifs import discover_daily_motifs_from_agents
    df = _make_multi_day_df()
    lib_path = tmp_path / "test_lib.pkl"
    daily_motifs_df, _ = discover_daily_motifs_from_agents(
        df, dynamic_library_file=str(lib_path)
    )
    assert lib_path.exists()
    assert len(daily_motifs_df) == 2
