"""The examples in the docstrings of the fitting and flow-comparison API must run."""

from __future__ import annotations

import doctest
import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "fastmob.measures.fitting.distance_distribution",
        "fastmob.measures.evaluation.flow_models",
        "fastmob.models.gravity",
    ],
)
def test_docstring_examples_run(module):
    if module.endswith("gravity"):
        pytest.importorskip("statsmodels")
    result = doctest.testmod(
        importlib.import_module(module), optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )

    assert result.attempted > 0
    assert result.failed == 0
