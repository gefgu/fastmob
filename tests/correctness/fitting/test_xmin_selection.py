"""Correctness tests for the automatic selection of xmin."""

from __future__ import annotations

import numpy as np
import pytest
from fastmob.measures.fitting import select_xmin


def test_select_xmin_finds_the_start_of_a_power_law_tail():
    rng = np.random.default_rng(4)
    tail = 10.0 * (rng.pareto(1.5, size=4000) + 1.0)  # a power law from 10 upward
    body = rng.uniform(1.0, 10.0, size=3000)  # something else below it
    table = select_xmin(np.concatenate([tail, body]), "powerlaw", n_candidates=25).to_pydict()

    assert table["selected"].count(True) == 1
    chosen = table["xmin"][table["selected"].index(True)]
    assert 5.0 <= chosen <= 30.0
    assert table["beta"][table["selected"].index(True)] == pytest.approx(2.5, abs=0.15)


def test_select_xmin_uses_given_candidates_and_checks_arguments():
    values = np.random.default_rng(1).pareto(1.5, size=2000) + 1.0

    table = select_xmin(values, "powerlaw", candidates=[1.0, 2.0, 4.0, 1e6])

    assert table["xmin"].to_pylist() == [1.0, 2.0, 4.0]  # the last leaves too few values
    with pytest.raises(ValueError, match="min_values"):
        select_xmin(values, min_values=5)
    with pytest.raises(ValueError, match="Too few"):
        select_xmin(values[:50], min_values=100)
