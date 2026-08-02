"""Correctness tests for FlowDataFrame methods added in the docs-parity refactor."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_fdf():
    """Small synthetic FlowDataFrame for testing."""
    from fastmob import FlowDataFrame

    flows = pd.DataFrame(
        {
            "origin": ["A", "A", "B", "B"],
            "destination": ["A", "B", "A", "B"],
            "flow": [100, 50, 30, 10],
        }
    )
    return FlowDataFrame(flows)


# ------------------------------------------------------------------
# get_flow
# ------------------------------------------------------------------


def test_get_flow_existing_pair():
    fdf = _make_fdf()
    assert fdf.get_flow("A", "B") == 50


def test_get_flow_self_loop():
    fdf = _make_fdf()
    assert fdf.get_flow("A", "A") == 100


def test_get_flow_missing_pair_returns_zero():
    fdf = _make_fdf()
    assert fdf.get_flow("A", "Z") == 0
    assert fdf.get_flow("Z", "A") == 0


# ------------------------------------------------------------------
# settings_from
# ------------------------------------------------------------------


def test_settings_from_copies_parameters():
    from fastmob import FlowDataFrame

    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [10]})
    fdf1 = FlowDataFrame(flows.copy(), parameters={"year": 2020})
    fdf2 = FlowDataFrame(flows.copy(), parameters={})
    fdf2.settings_from(fdf1)
    assert fdf2.parameters == {"year": 2020}


def test_settings_from_copies_tile_id():
    from fastmob import FlowDataFrame

    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [10]})
    fdf1 = FlowDataFrame(flows.copy(), tile_id="custom_tile")
    fdf2 = FlowDataFrame(flows.copy())
    fdf2.settings_from(fdf1)
    assert fdf2.tile_id == "custom_tile"


# ------------------------------------------------------------------
# to_matrix
# ------------------------------------------------------------------


def test_to_matrix_shape_and_values():
    fdf = _make_fdf()
    matrix = fdf.to_matrix()
    assert matrix.shape == (2, 2)  # tiles A and B
    # sorted tiles: A=0, B=1
    assert matrix[0, 0] == 100  # A→A
    assert matrix[0, 1] == 50  # A→B
    assert matrix[1, 0] == 30  # B→A
    assert matrix[1, 1] == 10  # B→B


def test_to_matrix_empty():
    from fastmob import FlowDataFrame

    flows = pd.DataFrame({"origin": [], "destination": [], "flow": []})
    fdf = FlowDataFrame(flows)
    matrix = fdf.to_matrix()
    assert matrix.shape == (0, 0)


# ------------------------------------------------------------------
# get_geometry (requires geopandas)
# ------------------------------------------------------------------


def test_get_geometry_returns_correct_shape():
    gpd = pytest.importorskip("geopandas", reason="geopandas required")
    from fastmob import FlowDataFrame
    from shapely.geometry import Point

    # Build a minimal tessellation
    tess = gpd.GeoDataFrame(
        {"tile_id": ["A", "B"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:4326",
    )
    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [5]})
    fdf = FlowDataFrame(flows, tessellation=tess)

    geom = fdf.get_geometry("A")
    assert geom.geom_type == "Point"


def test_get_geometry_missing_tile_raises():
    gpd = pytest.importorskip("geopandas", reason="geopandas required")
    from fastmob import FlowDataFrame
    from shapely.geometry import Point

    tess = gpd.GeoDataFrame(
        {"tile_id": ["A"], "geometry": [Point(0, 0)]},
        crs="EPSG:4326",
    )
    flows = pd.DataFrame({"origin": ["A"], "destination": ["A"], "flow": [5]})
    fdf = FlowDataFrame(flows, tessellation=tess)

    with pytest.raises(ValueError, match="not present"):
        fdf.get_geometry("Z")


def test_get_geometry_no_tessellation_raises():
    from fastmob import FlowDataFrame

    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [5]})
    fdf = FlowDataFrame(flows)

    with pytest.raises(ValueError, match="No tessellation"):
        fdf.get_geometry("A")


# ------------------------------------------------------------------
# plot methods (require folium)
# ------------------------------------------------------------------


def test_plot_flows_missing_dep(monkeypatch):
    """plot_flows raises ImportError when fastmob-vis is absent."""
    import sys

    monkeypatch.setitem(sys.modules, "fastmob_vis", None)

    from fastmob import FlowDataFrame

    flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [5]})
    fdf = FlowDataFrame(flows)

    with pytest.raises(ImportError, match="fastmob-vis"):
        fdf.plot_flows()
