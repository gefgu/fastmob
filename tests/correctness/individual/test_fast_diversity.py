"""Tests for the fast_diversity low-level primitive."""

import pytest

pydivsufsort = pytest.importorskip(
    "pydivsufsort",
    reason="pydivsufsort not installed; install with: pip install fastmob[diversity]",
)

from fastmob.measures.individual.fast_diversity import fast_diversity  # noqa: E402


def test_fast_diversity_constant_sequence_lower_than_varied():
    """A constant sequence has lower diversity than a varied one of the same length.

    Note: a constant sequence still has distinct substrings of different lengths
    (e.g., "A", "AA", "AAA", "AAAA") so diversity is not 0.0. It is, however,
    strictly lower than a fully unique sequence of the same length.
    """
    constant = fast_diversity(["A", "A", "A", "A"])
    varied = fast_diversity(["A", "B", "C", "D"])
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
    varied = ["A", "B", "C", "A", "B"]
    assert fast_diversity(uniform) < fast_diversity(varied)


def test_fast_diversity_raises_without_pydivsufsort(monkeypatch):
    """ImportError raised when pydivsufsort is None."""
    import sys
    import importlib

    # Ensure the submodule is loaded, then grab it directly from sys.modules
    # to avoid the name collision with the re-exported function in visits/__init__.py
    importlib.import_module("fastmob.measures.individual.fast_diversity")
    mod = sys.modules["fastmob.measures.individual.fast_diversity"]
    orig_div = mod._divsufsort
    orig_kasai = mod._kasai
    monkeypatch.setattr(mod, "_divsufsort", None)
    monkeypatch.setattr(mod, "_kasai", None)
    with pytest.raises(ImportError, match="pydivsufsort"):
        mod.fast_diversity(["A", "B"])
    monkeypatch.setattr(mod, "_divsufsort", orig_div)
    monkeypatch.setattr(mod, "_kasai", orig_kasai)
