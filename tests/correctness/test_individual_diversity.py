"""Tests for diversity measure (Step 7)."""
import pandas as pd
import numpy as np
import pytest


pydivsufsort = pytest.importorskip(
    "pydivsufsort",
    reason="pydivsufsort not installed; install with: pip install skmob2[diversity]",
)


from skmob2.measures.individual import fast_diversity, diversity


# ---------------------------------------------------------------------------
# fast_diversity unit tests
# ---------------------------------------------------------------------------

def test_fast_diversity_constant_sequence_lower_than_varied():
    """A constant sequence has lower diversity than a varied one of the same length.

    Note: a constant sequence still has distinct substrings of different lengths
    (e.g., "A", "AA", "AAA", "AAAA") so diversity is not 0.0. It is, however,
    strictly lower than a fully unique sequence of the same length.
    """
    constant = fast_diversity(["A", "A", "A", "A"])
    varied   = fast_diversity(["A", "B", "C", "D"])
    assert constant < varied


def test_fast_diversity_fully_unique_sequence_close_to_one():
    """A sequence of all distinct elements has high diversity."""
    n = 10
    seq = list(range(n))  # all unique
    result = fast_diversity(seq)
    # Not necessarily exactly 1.0, but should be well above 0
    assert result > 0.5


def test_fast_diversity_returns_float():
    seq = ["A", "B", "A", "B"]
    result = fast_diversity(seq)
    assert isinstance(result, float)


def test_fast_diversity_range():
    """fast_diversity must be in [0, 1] for any sequence."""
    sequences = [
        ["A", "A", "A"],
        ["A", "B", "C"],
        ["A", "B", "A", "B"],
        list(range(20)),
        [1, 1, 2, 1, 3, 2, 1],
    ]
    for seq in sequences:
        result = fast_diversity(seq)
        assert 0.0 <= result <= 1.0, f"Out of range for {seq}: {result}"


def test_fast_diversity_single_element():
    """A single-element sequence: 0 distinct substrings of length > 1."""
    result = fast_diversity(["X"])
    assert isinstance(result, float)
    # Degenerate case — at least should not crash


def test_fast_diversity_two_elements():
    seq_same = ["A", "A"]
    seq_diff = ["A", "B"]
    r_same = fast_diversity(seq_same)
    r_diff = fast_diversity(seq_diff)
    # Different sequence should have higher or equal diversity
    assert r_diff >= r_same


def test_fast_diversity_longer_unique_beats_short_unique():
    """Longer uniform sequence has lower diversity than varied one of same length."""
    uniform = ["A", "A", "A", "A", "A"]
    varied  = ["A", "B", "C", "A", "B"]
    assert fast_diversity(uniform) < fast_diversity(varied)


def test_fast_diversity_raises_without_pydivsufsort(monkeypatch):
    """ImportError raised when pydivsufsort is None."""
    import skmob2.measures.individual as mod
    orig_div = mod._divsufsort
    orig_kasai = mod._kasai
    monkeypatch.setattr(mod, "_divsufsort", None)
    monkeypatch.setattr(mod, "_kasai", None)
    with pytest.raises(ImportError, match="pydivsufsort"):
        mod.fast_diversity(["A", "B"])
    monkeypatch.setattr(mod, "_divsufsort", orig_div)
    monkeypatch.setattr(mod, "_kasai", orig_kasai)


# ---------------------------------------------------------------------------
# diversity (DataFrame-level) tests
# ---------------------------------------------------------------------------

def _diversity_visits():
    """Two users with predictable diversity patterns.

    u1: alternating A/B → moderate diversity
    u2: always same → low diversity (close to 0)
    """
    return pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1", "u1",
                        "u2", "u2", "u2", "u2"],
        "location_id": ["home", "work", "home", "work",
                        "home", "home", "home", "home"],
        "location_type": ["HOME", "WORK", "HOME", "WORK",
                          "HOME", "HOME", "HOME", "HOME"],
    })


def test_diversity_result_shape():
    """Result has one row per user and columns [user_id_col, 'diversity']."""
    df = _diversity_visits()
    result = diversity(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "agent_id" in result.columns
    assert "diversity" in result.columns


def test_diversity_range():
    """diversity values must be in [0, 1]."""
    df = _diversity_visits()
    result = diversity(df)
    assert (result["diversity"] >= 0).all()
    assert (result["diversity"] <= 1.0 + 1e-9).all()


def test_diversity_constant_user_has_lower_diversity_than_varied():
    """User who only visits one location has lower diversity than one who varies.

    A constant sequence still has some distinct substrings (same location_key
    repeated gives "AA", "AAA", etc.), so diversity is not exactly 0.
    However, it is strictly less than a fully varied sequence.
    """
    df_const = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1", "u1"],
        "location_id": ["home", "home", "home", "home"],
        "location_type": ["HOME", "HOME", "HOME", "HOME"],
    })
    df_varied = pd.DataFrame({
        "agent_id":    ["u2", "u2", "u2", "u2"],
        "location_id": ["a", "b", "c", "d"],
        "location_type": ["T1", "T2", "T3", "T4"],
    })
    r_const = diversity(df_const).iloc[0]["diversity"]
    r_varied = diversity(df_varied).iloc[0]["diversity"]
    assert r_const < r_varied


def test_diversity_varied_user_beats_constant_user():
    """More varied visit sequence produces higher diversity."""
    df = _diversity_visits()
    result = diversity(df).set_index("agent_id")
    # u2 is constant (all home_HOME) → diversity close to 0
    # u1 alternates → higher diversity
    assert result.loc["u1", "diversity"] > result.loc["u2", "diversity"]


def test_diversity_without_location_type():
    """When location_type_col is None, factorize on location_id alone."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1", "u1"],
        "location_id": ["home", "work", "home", "work"],
    })
    result = diversity(df, location_type_col=None)
    assert "diversity" in result.columns
    assert len(result) == 1


def test_diversity_column_autodetection_user_id():
    """Auto-detects 'user_id' when 'agent_id' is absent."""
    df = pd.DataFrame({
        "user_id":     ["u1", "u1", "u2", "u2"],
        "location_id": ["a", "b", "c", "c"],
        "location_type": ["T1", "T2", "T3", "T3"],
    })
    result = diversity(df)
    assert "user_id" in result.columns
    assert len(result) == 2


def test_diversity_column_autodetection_purpose():
    """Auto-detects 'purpose' as location_type_col."""
    df = pd.DataFrame({
        "agent_id":  ["u1", "u1"],
        "location_id": ["home", "work"],
        "purpose":   ["HOME", "WORK"],
    })
    result = diversity(df)
    assert "diversity" in result.columns
    assert len(result) == 1


def test_diversity_multi_user_count():
    """Returns exactly one row per user."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u2", "u3", "u1", "u2"],
        "location_id": ["a", "b", "c", "a", "a"],
        "location_type": ["T1", "T2", "T3", "T1", "T1"],
    })
    result = diversity(df)
    assert len(result) == 3
    assert set(result["agent_id"]) == {"u1", "u2", "u3"}
