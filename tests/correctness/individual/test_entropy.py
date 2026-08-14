"""Tests for trajectory entropy and predictability measures."""

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# trajectory_entropy DataFrame tests
# ---------------------------------------------------------------------------


def _entropy_visits():
    """Two users: u1 with repeated pattern, u2 with varied pattern."""
    return pd.DataFrame(
        {
            "agent_id": ["u1"] * 6 + ["u2"] * 6,
            "location_id": ["h", "w", "h", "w", "h", "w", "a", "b", "c", "d", "e", "f"],
            "location_type": ["HOME", "WORK"] * 3 + ["T1", "T2", "T3", "T4", "T5", "T6"],
        }
    )


def test_trajectory_entropy_result_shape():
    """Result has one row per user and columns [user_id_col, 'entropy']."""
    from fastmob.measures.individual.entropy import trajectory_entropy

    df = _entropy_visits()
    result = trajectory_entropy(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "agent_id" in result.columns
    assert "entropy" in result.columns


def test_trajectory_entropy_normalized_in_range():
    """Normalized entropy must be in [0, 1]."""
    from fastmob.measures.individual.entropy import trajectory_entropy

    df = _entropy_visits()
    result = trajectory_entropy(df, normalized=True)
    assert (result["entropy"] >= 0).all()
    assert (result["entropy"] <= 1.0 + 1e-9).all()


def test_trajectory_entropy_constant_lower_than_varied():
    """Constant visit sequence has lower entropy than a varied one.

    Note: the Kontoyiannis estimator has finite-sample variance and does not
    return exactly 0.0 for constant sequences when n is small. The invariant
    is that constant < varied.
    """
    from fastmob.measures.individual.entropy import trajectory_entropy

    df_const = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1", "u1"],
            "location_id": ["home", "home", "home", "home"],
            "location_type": ["HOME", "HOME", "HOME", "HOME"],
        }
    )
    df_varied = pd.DataFrame(
        {
            "agent_id": ["u2", "u2", "u2", "u2"],
            "location_id": ["a", "b", "c", "d"],
            "location_type": ["T1", "T2", "T3", "T4"],
        }
    )
    r_const = trajectory_entropy(df_const, normalized=True).iloc[0]["entropy"]
    r_varied = trajectory_entropy(df_varied, normalized=True).iloc[0]["entropy"]
    assert r_const < r_varied


def test_trajectory_entropy_varied_higher_than_repeated():
    """More varied sequence has higher normalized entropy."""
    from fastmob.measures.individual.entropy import trajectory_entropy

    df = _entropy_visits()
    result = trajectory_entropy(df).set_index("agent_id")
    # u2 has 6 distinct locations (all different) → higher entropy
    # u1 alternates between 2 locations → lower entropy
    assert result.loc["u2", "entropy"] >= result.loc["u1", "entropy"]


def test_trajectory_entropy_column_autodetection():
    """Auto-detects 'user_id' and 'purpose' columns."""
    from fastmob.measures.individual.entropy import trajectory_entropy

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "location_id": ["a", "a", "b", "c"],
            "purpose": ["HOME", "HOME", "WORK", "LEISURE"],
        }
    )
    result = trajectory_entropy(df)
    assert "user_id" in result.columns
    assert len(result) == 2


# ---------------------------------------------------------------------------
# trajectory_predictability DataFrame tests
# ---------------------------------------------------------------------------


def test_trajectory_predictability_result_columns():
    """Result has columns [user_id_col, 'real_entropy', 'predictability',
    'n_unique_locations', 'n_steps']."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = _entropy_visits()
    result = trajectory_predictability(df)
    expected_cols = {"agent_id", "real_entropy", "predictability", "n_unique_locations", "n_steps"}
    assert expected_cols.issubset(set(result.columns))


def test_trajectory_predictability_one_row_per_user():
    """One row per user."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = _entropy_visits()
    result = trajectory_predictability(df)
    assert len(result) == 2


def test_trajectory_predictability_range():
    """Predictability must be in [0, 1]; n_steps == row count per user."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = _entropy_visits()
    result = trajectory_predictability(df)
    assert (result["predictability"] >= 0).all()
    assert (result["predictability"] <= 1.0 + 1e-9).all()
    # Each user has 6 rows
    assert (result["n_steps"] == 6).all()


def test_trajectory_predictability_constant_user_is_fully_predictable():
    """User who always visits same place → predictability close to 1."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1", "u1"],
            "location_id": ["home", "home", "home", "home"],
            "location_type": ["HOME", "HOME", "HOME", "HOME"],
        }
    )
    result = trajectory_predictability(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    assert row["predictability"] == pytest.approx(1.0, abs=1e-6)
    assert row["n_unique_locations"] == 1
    assert row["n_steps"] == 4


def test_trajectory_predictability_n_unique_locations_correct():
    """n_unique_locations matches the count of distinct location tokens."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u1", "u1", "u1"],
            "location_id": ["a", "b", "a", "c"],
            "location_type": ["T1", "T2", "T1", "T3"],
        }
    )
    result = trajectory_predictability(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # Distinct tokens: "a_T1", "b_T2", "c_T3" → 3
    assert row["n_unique_locations"] == 3


def test_trajectory_predictability_column_autodetection():
    """Auto-detects 'user_id' column."""
    from fastmob.measures.individual.entropy import trajectory_predictability

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "location_id": ["a", "a", "b", "c"],
            "location_type": ["HOME", "HOME", "WORK", "LEISURE"],
        }
    )
    result = trajectory_predictability(df)
    assert "user_id" in result.columns
    assert len(result) == 2
