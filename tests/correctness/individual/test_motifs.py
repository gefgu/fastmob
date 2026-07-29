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


def test_discover_motifs_handles_unsorted_input():
    """The indexed kernel path must preserve chronological per-user visits."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    df = pd.concat(
        [
            df,
            df.assign(
                agent_id="u2",
                start_timestamp=df["start_timestamp"] + pd.Timedelta(hours=1),
                end_timestamp=df["end_timestamp"] + pd.Timedelta(hours=1),
            ),
        ],
        ignore_index=True,
    )
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    expected, _ = discover_daily_motifs_from_agents(df)
    actual, _ = discover_daily_motifs_from_agents(shuffled)

    sort_cols = ["agent_id", "date"]
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = actual.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_discover_motifs_distribution_sum_to_100():
    """Motif distribution percentages sum to 100."""
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    _, motif_dist_df = discover_daily_motifs_from_agents(df)
    assert motif_dist_df["percentage"].sum() == pytest.approx(100.0, abs=1e-6)


def test_discover_motifs_preserves_numeric_user_id_dtype():
    """A numeric agent_id source column must come back numeric, not string.

    The Rust kernel never sees the uid at all (it only tags each result
    with a position into the per-user ranges); the wrapper must gather the
    matching label back in its original dtype rather than leaking whatever
    intermediate representation was used to build the ranges.
    """
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    df["agent_id"] = df["agent_id"].map({"u1": 1}).astype("int64")

    daily_motifs_df, _ = discover_daily_motifs_from_agents(df)
    assert pd.api.types.is_integer_dtype(daily_motifs_df["agent_id"])
    assert set(daily_motifs_df["agent_id"].unique()) == {1}


def test_discover_motifs_preserves_numeric_user_id_dtype_polars():
    """Same numeric-uid dtype guarantee on the Polars backend.

    Regression test: an earlier version of the vectorized uid gather built
    an intermediate ``dtype=object`` NumPy array, which Polars represents
    as an ``Object`` column that cannot be cast back to Int64.
    """
    pl = pytest.importorskip("polars")
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    df["agent_id"] = df["agent_id"].map({"u1": 1}).astype("int64")
    pldf = pl.from_pandas(df)

    daily_motifs_df, _ = discover_daily_motifs_from_agents(pldf)
    assert daily_motifs_df.schema["agent_id"] == pl.Int64
    assert set(daily_motifs_df["agent_id"].unique().to_list()) == {1}


def test_discover_motifs_handles_tz_aware_timestamps():
    """Tz-aware start/end timestamps (e.g. UTC-stamped check-ins) must not raise.

    Regression test: casting a tz-aware datetime column straight to a naive
    Datetime dtype previously raised a pandas TypeError.
    """
    from fastmob.measures.individual.motifs import discover_daily_motifs_from_agents

    df = _make_multi_day_df()
    df["start_timestamp"] = df["start_timestamp"].dt.tz_localize("UTC")
    df["end_timestamp"] = df["end_timestamp"].dt.tz_localize("UTC")

    naive_df = _make_multi_day_df()
    tz_daily_df, _ = discover_daily_motifs_from_agents(df)
    naive_daily_df, _ = discover_daily_motifs_from_agents(naive_df)

    assert tz_daily_df["motif_id"].tolist() == naive_daily_df["motif_id"].tolist()
    assert tz_daily_df["date"].tolist() == naive_daily_df["date"].tolist()
