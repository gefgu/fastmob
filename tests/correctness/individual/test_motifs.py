"""Tests for motif classification measures."""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_multi_day_df():
    """Two days of visits for one user.  Each day: HOME→WORK→HOME."""
    rows = []
    for day_offset in range(2):
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_offset)
        rows += [
            {
                "agent_id": "u1",
                "location_id": "home",
                "purpose": "HOME",
                "start_timestamp": base + pd.Timedelta(hours=0),
                "end_timestamp": base + pd.Timedelta(hours=8),
                "duration_minutes": 480,
            },
            {
                "agent_id": "u1",
                "location_id": "work",
                "purpose": "WORK",
                "start_timestamp": base + pd.Timedelta(hours=9),
                "end_timestamp": base + pd.Timedelta(hours=17),
                "duration_minutes": 480,
            },
            {
                "agent_id": "u1",
                "location_id": "home",
                "purpose": "HOME",
                "start_timestamp": base + pd.Timedelta(hours=18),
                "end_timestamp": base + pd.Timedelta(hours=23),
                "duration_minutes": 300,
            },
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# discover_daily_motifs_from_agents tests
# ---------------------------------------------------------------------------


def test_discover_motifs_returns_two_dataframes():
    """discover_daily_motifs_from_agents returns exactly two DataFrames."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    result = discover_daily_motifs_from_agents(df)
    assert len(result) == 2
    daily_motifs_df, motif_dist_df = result
    assert isinstance(daily_motifs_df, pd.DataFrame)
    assert isinstance(motif_dist_df, pd.DataFrame)


def test_discover_motifs_daily_rows_count():
    """Two days, one user → daily_motifs_df has 2 rows."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    assert len(daily_motifs_df) == 2


def test_discover_motifs_has_required_columns():
    """daily_motifs_df has required columns."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    required = {"agent_id", "date", "motif_id", "num_nodes", "num_edges"}
    assert required.issubset(set(daily_motifs_df.columns))


def test_discover_motifs_home_work_home_classified_correctly():
    """HOME→WORK→HOME each day → simple return motif (2 nodes, 2 edges)."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    # All days are simple returns: 2 nodes, 2 edges
    assert (daily_motifs_df["num_nodes"] == 2).all()
    assert (daily_motifs_df["num_edges"] == 2).all()


def test_discover_motifs_distribution_sum_to_100():
    """Motif distribution percentages sum to 100."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    _, motif_dist_df = discover_daily_motifs_from_agents(df)
    assert motif_dist_df["percentage"].sum() == pytest.approx(100.0, abs=1e-6)
