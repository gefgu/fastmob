"""The examples in the docstrings must run."""

from __future__ import annotations

import doctest
import importlib

import pytest


@pytest.mark.parametrize("module", ["charts", "figure", "motifs"])
def test_docstring_examples_run(module):
    result = doctest.testmod(
        importlib.import_module(f"fastmob_vis.{module}"),
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )

    assert result.failed == 0
    assert result.attempted > 0
