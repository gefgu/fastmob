"""Geo helpers used by tessellation tilers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _constants as constants


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
        raise ImportError("geopandas is required for tessellation: pip install skmob2[tessellation]") from exc
    return gpd


def _require_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
        raise ImportError("requests is required for string place tessellation: pip install skmob2[tessellation]") from exc
    return requests


def _require_shapely_geometry():
    try:
        import shapely.geometry as geometry
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
        raise ImportError("shapely is required for tessellation: pip install skmob2[tessellation]") from exc
    return geometry


def bbox_from_points(points: Any, crs=None):
    """Build a GeoDataFrame bounding box around a point collection or bounds."""
    gpd = _require_geopandas()
    geometry = _require_shapely_geometry()

    try:
        coords = points.total_bounds
    except AttributeError:
        coords = points

    base = geometry.box(coords[0], coords[1], coords[2], coords[3], ccw=True)
    base = gpd.GeoDataFrame(geometry=[base], crs=constants.DEFAULT_CRS)
    if crs is None:
        return base
    return base.to_crs(crs)


def bbox_from_name(query: str, which_osm_result: int = 0, crs=None):
    """Create a GeoDataFrame from an OpenStreetMap place-name query."""
    gpd = _require_geopandas()
    geometry = _require_shapely_geometry()
    requests = _require_requests()

    nominatim_url = "https://nominatim.openstreetmap.org/search.php?q=%s&polygon_geojson=1&format=json" % query
    response = requests.get(nominatim_url, timeout=30)
    response.raise_for_status()
    data = response.json()

    features = []
    for result in data:
        bbox_south, bbox_north, bbox_west, bbox_east = [float(x) for x in result["boundingbox"]]
        coords = result["geojson"]["coordinates"]
        try:
            geom = geometry.MultiPolygon([geometry.Polygon(c[0], [inner for inner in c[1:]]) for c in coords])
        except TypeError:
            try:
                geom = geometry.MultiPolygon([geometry.Polygon(coords[0], [inner for inner in coords[1:]])])
            except TypeError:
                geom = result["geojson"]

        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "place_name": result["display_name"],
                    "bbox_north": bbox_north,
                    "bbox_south": bbox_south,
                    "bbox_east": bbox_east,
                    "bbox_west": bbox_west,
                },
            }
        )

    gdf = gpd.GeoDataFrame.from_features(features) if features else gpd.GeoDataFrame()
    if crs is None:
        gdf.crs = constants.DEFAULT_CRS
    else:
        gdf = gdf.to_crs(crs)
    if which_osm_result >= 0 and len(gdf) > 0:
        gdf = gdf.loc[[which_osm_result]]
    return gdf


def nearest(origin, tessellation, col: str):
    """Return values from ``col`` in the nearest tessellation point for each origin point."""
    from skmob2.models._common import haversine_km

    def _nearest(row, points):
        origin_lat_lng = (row["geometry"].y, row["geometry"].x)
        near = float("+inf")
        point = None
        for index, point_row in points.iterrows():
            point_geometry = point_row["geometry"]
            distance = haversine_km(origin_lat_lng, (point_geometry.y, point_geometry.x))
            if distance < near:
                near = distance
                point = index
        return point

    return tessellation.iloc[origin.apply(_nearest, args=(tessellation,), axis=1)][col]


def get_geom_centroid(geom, return_lat_lng: bool = False) -> list[float]:
    """Compute the centroid coordinates of a shapely geometry."""
    x, y = geom.centroid.xy
    lng = float(np.asarray(x)[0])
    lat = float(np.asarray(y)[0])
    if return_lat_lng:
        return [lat, lng]
    return [lng, lat]
