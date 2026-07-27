"""Tests for trajectory entropy and predictability measures."""

import math

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# _kontoyiannis_entropy unit tests
# ---------------------------------------------------------------------------


def test_kontoyiannis_entropy_constant_lower_than_varied():
    """Constant sequence has lower entropy than a fully varied sequence.

    Note: for short sequences the Kontoyiannis estimator has variance and
    does not return exactly 0 for constant sequences. The key property is
    that constant < varied.
    """
    from fastmob.measures.individual.entropy import _kontoyiannis_entropy

    constant = _kontoyiannis_entropy(["A", "A", "A", "A"])
    varied = _kontoyiannis_entropy(["A", "B", "C", "D"])
    assert constant < varied


def test_kontoyiannis_entropy_returns_nonnegative():
    """Entropy must be non-negative."""
    from fastmob.measures.individual.entropy import _kontoyiannis_entropy

    sequences = [
        ["A", "B", "A", "B"],
        ["A", "B", "C", "D"],
        ["X"],
        ["A", "A", "B", "A", "C", "B"],
    ]
    for seq in sequences:
        result = _kontoyiannis_entropy(seq)
        assert result >= 0.0, f"Negative entropy for {seq}: {result}"


def test_kontoyiannis_entropy_single_element_sequence():
    """Length-1 sequence returns 0."""
    from fastmob.measures.individual.entropy import _kontoyiannis_entropy

    result = _kontoyiannis_entropy(["A"])
    assert result == pytest.approx(0.0, abs=1e-9)


def test_kontoyiannis_entropy_abab_in_range():
    """ABAB entropy is in [0, log2(4)] ≈ [0, 2.0]."""
    from fastmob.measures.individual.entropy import _kontoyiannis_entropy

    result = _kontoyiannis_entropy(["A", "B", "A", "B"])
    assert 0.0 <= result <= math.log2(4) + 1e-9


def test_kontoyiannis_entropy_varied_higher_than_repeated():
    """More varied sequence has higher entropy than repeated pattern."""
    from fastmob.measures.individual.entropy import _kontoyiannis_entropy

    repeated = _kontoyiannis_entropy(["A", "B", "A", "B", "A", "B"])
    varied = _kontoyiannis_entropy(["A", "B", "C", "D", "E", "F"])
    assert varied >= repeated


# ---------------------------------------------------------------------------
# _fano_equation_term and _solve_max_predictability_with_fano unit tests
# ---------------------------------------------------------------------------


def test_fano_solve_zero_entropy_returns_one():
    """Zero real entropy → max predictability = 1.0."""
    from fastmob.measures.individual.entropy import _solve_max_predictability_with_fano

    result = _solve_max_predictability_with_fano(real_entropy=0.0, n_unique=5)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_fano_solve_max_entropy_returns_one_over_n():
    """Max entropy (log2(n)) → max predictability = 1/n."""
    from fastmob.measures.individual.entropy import _solve_max_predictability_with_fano

    n = 5
    max_entropy = math.log2(n)
    result = _solve_max_predictability_with_fano(real_entropy=max_entropy, n_unique=n)
    assert result == pytest.approx(1.0 / n, abs=1e-6)


def test_fano_solve_single_unique_location():
    """n_unique=1 → predictability = 1.0 (trivially predictable)."""
    from fastmob.measures.individual.entropy import _solve_max_predictability_with_fano

    result = _solve_max_predictability_with_fano(real_entropy=0.0, n_unique=1)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_fano_solve_predictability_in_range():
    """Predictability must be in [1/n, 1] for any entropy in [0, log2(n)]."""
    from fastmob.measures.individual.entropy import _solve_max_predictability_with_fano

    n = 8
    for entropy in [0.0, 0.5, 1.0, 2.0, math.log2(n)]:
        p = _solve_max_predictability_with_fano(real_entropy=entropy, n_unique=n)
        assert 1.0 / n - 1e-6 <= p <= 1.0 + 1e-6, f"Out of range for entropy={entropy}: predictability={p}"


def test_fano_solve_monotone_decreasing_in_entropy():
    """Higher entropy → lower max predictability."""
    from fastmob.measures.individual.entropy import _solve_max_predictability_with_fano

    n = 6
    entropies = [0.0, 0.5, 1.0, 1.5, math.log2(n)]
    predictabilities = [_solve_max_predictability_with_fano(e, n) for e in entropies]
    for i in range(len(predictabilities) - 1):
        assert predictabilities[i] >= predictabilities[i + 1] - 1e-9


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
