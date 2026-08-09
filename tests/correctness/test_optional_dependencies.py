"""User-facing errors for optional, feature-local integrations."""

from __future__ import annotations

import sys

import pytest


def test_network_fetch_explains_how_to_install_duckdb(monkeypatch):
    from fastmob.network import fetch_road_network

    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"pip install duckdb"):
        fetch_road_network(0.0, 0.0, 1.0, 1.0, "2026-05-20.0")


def test_gravity_fit_explains_how_to_install_statsmodels(monkeypatch):
    from fastmob.models.gravity import Gravity

    monkeypatch.setitem(sys.modules, "statsmodels", None)
    with pytest.raises(ImportError, match=r"pip install statsmodels"):
        Gravity().fit(None)
