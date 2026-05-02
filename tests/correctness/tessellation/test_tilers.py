"""Correctness tests for skmob2.tessellation.tilers."""

from __future__ import annotations

import pytest

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
pytest.importorskip("h3")

from skmob2.tessellation import tilers  # noqa: E402


@pytest.fixture()
def bbox():
    poly = [
        [
            [116.1440758191, 39.8846396072],
            [116.3449987678, 39.8846396072],
            [116.3449987678, 40.0430521004],
            [116.1440758191, 40.0430521004],
            [116.1440758191, 39.8846396072],
        ]
    ]
    geom = [shapely_geometry.Polygon(p) for p in poly]
    return gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")


def test_tessellation_imports():
    assert tilers.tiler is not None
    assert "squared" in tilers.tiler._tilers
    assert "voronoi" in tilers.tiler._tilers
    assert "h3_tessellation" in tilers.tiler._tilers


def test_tiler_invalid_key_raises():
    with pytest.raises(ValueError, match="missing"):
        tilers.tiler.get("missing")


def test_voronoi_from_points_uses_tile_id():
    points = gpd.GeoDataFrame(
        geometry=[
            shapely_geometry.Point(0.0, 0.0),
            shapely_geometry.Point(1.0, 1.0),
        ],
        crs="EPSG:4326",
    )

    tessellation = tilers.tiler.get("voronoi", points=points)

    assert list(tessellation.columns) == ["tile_id", "geometry"]
    assert tessellation["tile_id"].tolist() == ["0", "1"]
    assert all(isinstance(x, shapely_geometry.Point) for x in tessellation.geometry)


def test_voronoi_can_emit_legacy_tile_id():
    points = gpd.GeoDataFrame(geometry=[shapely_geometry.Point(0.0, 0.0)], crs="EPSG:4326")

    tessellation = tilers.tiler.get("voronoi", points=points, tile_id_col="tile_ID")

    assert "tile_ID" in tessellation.columns
    assert "tile_id" not in tessellation.columns


def test_squared_from_polygon_uses_tile_id(bbox):
    tessellation = tilers.tiler.get("squared", base_shape=bbox, meters=15000)

    assert isinstance(tessellation, gpd.GeoDataFrame)
    assert "tile_id" in tessellation.columns
    assert len(tessellation) > 0
    assert all(isinstance(x, shapely_geometry.Polygon) for x in tessellation.geometry)


def test_squared_from_points_builds_bounding_grid():
    points = gpd.GeoDataFrame(
        geometry=[
            shapely_geometry.Point(116.1440758191, 39.8846396072),
            shapely_geometry.Point(116.3449987678, 40.0430521004),
        ],
        crs="EPSG:4326",
    )

    tessellation = tilers.tiler.get("squared", base_shape=points, meters=15000)

    assert "tile_id" in tessellation.columns
    assert len(tessellation) > 0


def test_h3_tessellation_from_polygon_uses_tile_id(bbox):
    tessellation = tilers.tiler.get("h3_tessellation", base_shape=bbox, meters=5000)

    assert isinstance(tessellation, gpd.GeoDataFrame)
    assert {"tile_id", "H3_INDEX", "geometry"}.issubset(tessellation.columns)
    assert len(tessellation) > 0


@pytest.mark.parametrize("input_meters, expected_resolution", [(50, 10), (500, 8), (5000, 6)])
def test_h3_meters_to_resolution(input_meters, expected_resolution):
    assert tilers.H3TessellationTiler()._meters_to_resolution(input_meters) == expected_resolution


def test_h3_get_resolution_matches_original_fixture(bbox):
    assert tilers.H3TessellationTiler()._get_resolution(base_shape=bbox, meters=5000) == 6
