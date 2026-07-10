"""Tessellation tilers ported from scikit-mobility."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from . import _constants as constants
from . import _utils as utils


def _require_geopandas():
    return utils._require_geopandas()


def _require_shapely():
    try:
        import shapely.geometry as geometry
        import shapely.ops as ops
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
        raise ImportError("shapely is required for tessellation: pip install fkmob[tessellation]") from exc
    return geometry, ops


def _require_h3():
    try:
        import h3.api.numpy_int as h3
    except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
        raise ImportError("h3 is required for H3 tessellation: pip install fkmob[tessellation]") from exc
    return h3


def _union_geometries(geometries):
    _geometry, ops = _require_shapely()
    union = getattr(ops, "cascaded_union", None)
    if union is None:
        union = ops.unary_union
    return union(geometries)


def _is_geometry_collection(value: Any) -> bool:
    gpd = _require_geopandas()
    return isinstance(value, (gpd.GeoDataFrame, gpd.GeoSeries))


class TessellationTilers:
    """Registry for tessellation tiler implementations."""

    def __init__(self):
        self._tilers = {}

    def register_tiler(self, key: str, tiler_instance):
        self._tilers[key] = tiler_instance

    def create(self, key: str, **kwargs):
        tiler_instance = self._tilers.get(key)
        if not tiler_instance:
            raise ValueError(key)
        return tiler_instance(**kwargs)

    def get(self, service_id: str, **kwargs):
        return self.create(service_id, **kwargs)


tiler = TessellationTilers()


class TessellationTiler(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def _build(self, **kwargs):
        pass


class VoronoiTessellationTiler(TessellationTiler):
    """Build a point tessellation from input points."""

    def __call__(self, points, crs=constants.DEFAULT_CRS, tile_id_col: str = constants.TILE_ID):
        gpd = _require_geopandas()
        geometry, _ops = _require_shapely()

        if isinstance(points, gpd.GeoDataFrame) and not all(isinstance(x, geometry.Point) for x in points.geometry):
            raise ValueError("Not valid points object. Accepted type is GeoDataFrame with Point geometries.")
        return self._build(points=points, crs=crs, tile_id_col=tile_id_col)

    def _build(self, points, crs=constants.DEFAULT_CRS, tile_id_col: str = constants.TILE_ID):
        gpd = _require_geopandas()

        gdf = gpd.GeoDataFrame(points.copy(), crs=crs)
        gdf.loc[:, tile_id_col] = list(np.arange(0, len(gdf)))
        gdf[tile_id_col] = gdf[tile_id_col].astype("str")
        return gdf[[tile_id_col, "geometry"]]


tiler.register_tiler("voronoi", VoronoiTessellationTiler())


class SquaredTessellationTiler(TessellationTiler):
    """Build a square grid over a base shape."""

    def __call__(
        self,
        base_shape,
        meters: float = 50,
        which_osm_result: int = -1,
        crs=constants.DEFAULT_CRS,
        window_size=None,
        tile_id_col: str = constants.TILE_ID,
    ):
        del window_size
        geometry, _ops = _require_shapely()

        if isinstance(base_shape, str):
            base_shape = self._str_to_first_polygon(base_shape, which_osm_result)
        elif _is_geometry_collection(base_shape):
            if all(isinstance(x, geometry.Point) for x in base_shape.geometry):
                base_shape = utils.bbox_from_points(base_shape, base_shape.crs)
            elif all(isinstance(x, (geometry.Polygon, geometry.MultiPolygon)) for x in base_shape.geometry):
                base_shape = self._merge_all_polygons(base_shape)
            else:
                raise ValueError("Not valid base_shape object. GeoDataFrame geometries must be points or polygons.")
        else:
            raise ValueError("Not valid base_shape object. Accepted types are str, GeoDataFrame or GeoSeries.")

        return self._build(base_shape=base_shape, meters=meters, crs=crs, tile_id_col=tile_id_col)

    def _str_to_first_polygon(self, base_shape: str, which_osm_result: int):
        geometry, _ops = _require_shapely()
        base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
        i = 0
        selected = base_shapes.loc[[i]]
        while not isinstance(selected.geometry.iloc[0], (geometry.Polygon, geometry.MultiPolygon)):
            i += 1
            selected = base_shapes.loc[[i]]
        return selected

    def _merge_all_polygons(self, base_shape):
        gpd = _require_geopandas()
        return gpd.GeoSeries(_union_geometries(base_shape.geometry.values), crs=base_shape.crs)

    def _build(self, base_shape, meters, crs=constants.DEFAULT_CRS, tile_id_col: str = constants.TILE_ID):
        gpd = _require_geopandas()
        geometry, _ops = _require_shapely()

        tmp_crs = constants.UNIVERSAL_CRS
        area = base_shape.to_crs(tmp_crs)
        min_x, min_y, max_x, max_y = area.total_bounds

        x_squares = int(math.ceil(math.fabs(max_x - min_x) / meters))
        y_squares = int(math.ceil(math.fabs(min_y - max_y) / meters))
        shape = area.union_all() if hasattr(area, "union_all") else area.unary_union
        polygons = []

        for i in range(0, x_squares):
            x1 = min_x + (meters * i)
            x2 = min_x + (meters * (i + 1))
            for j in range(0, y_squares):
                y1 = min_y + (meters * j)
                y2 = min_y + (meters * (j + 1))
                polygon = geometry.Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
                if shape.intersects(polygon):
                    polygons.append({"geometry": polygon})

        gdf = gpd.GeoDataFrame(polygons, crs=tmp_crs)
        gdf = gdf.reset_index().rename(columns={"index": tile_id_col})
        gdf[tile_id_col] = gdf[tile_id_col].astype("str")
        return gdf.to_crs(crs)


tiler.register_tiler("squared", SquaredTessellationTiler())


class H3TessellationTiler(TessellationTiler):
    """Build an H3 hexagonal tessellation over a base shape."""

    def __call__(
        self,
        base_shape,
        meters: float = 50,
        which_osm_result: int = -1,
        crs=constants.DEFAULT_CRS,
        window_size=None,
        tile_id_col: str = constants.TILE_ID,
    ):
        del window_size
        base_shape_geometry = self._create_geometry_if_does_not_exists(base_shape, which_osm_result)
        base_shape_geometry_merged = self._merge_all_polygons(base_shape_geometry)
        return self._build(base_shape=base_shape_geometry_merged, meters=meters, crs=crs, tile_id_col=tile_id_col)

    def _create_geometry_if_does_not_exists(self, base_shape, which_osm_result):
        geometry, _ops = _require_shapely()

        if isinstance(base_shape, str):
            return self._str_to_geometry(base_shape, which_osm_result)
        if self._isinstance_geodataframe_or_geoseries(base_shape):
            if all(isinstance(x, geometry.Point) for x in base_shape.geometry):
                return utils.bbox_from_points(base_shape, base_shape.crs)
            return base_shape
        raise ValueError("Not valid base_shape object. Accepted types are str, GeoDataFrame or GeoSeries.")

    def _isinstance_geodataframe_or_geoseries(self, base_shape):
        return _is_geometry_collection(base_shape)

    def _str_to_geometry(self, base_shape: str, which_osm_result: int):
        base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
        return self._find_first_polygon(base_shapes)

    def _find_first_polygon(self, base_shapes):
        return_shape = base_shapes.iloc[[0]]
        for i, current_shape in enumerate(base_shapes["geometry"].values):
            if self._isinstance_poly_or_multipolygon(current_shape):
                return_shape = base_shapes.iloc[[i]]
                break
        return return_shape

    def _isinstance_poly_or_multipolygon(self, shape):
        geometry, _ops = _require_shapely()
        return isinstance(shape, (geometry.Polygon, geometry.MultiPolygon))

    def _merge_all_polygons(self, base_shape):
        gpd = _require_geopandas()
        polygons = base_shape.geometry.values
        return gpd.GeoSeries(_union_geometries(polygons), crs=base_shape.crs)

    def _build(self, base_shape, meters, crs=constants.DEFAULT_CRS, tile_id_col: str = constants.TILE_ID):
        if base_shape.crs != constants.DEFAULT_CRS:
            base_shape = base_shape.to_crs(constants.DEFAULT_CRS)
        resolution = self._get_resolution(base_shape, meters)
        hexagon_ids = self._handle_polyfill(base_shape, resolution)
        hexagon_polygons = self._create_hexagon_polygons(hexagon_ids)
        hexagon_polygons = self._add_tile_id(hexagon_polygons, tile_id_col=tile_id_col)
        return hexagon_polygons.to_crs(crs)

    def _get_resolution(self, base_shape, meters):
        resolution = self._meters_to_resolution(meters)
        base_shape_projected = base_shape.to_crs(constants.UNIVERSAL_CRS)
        minimum_resolution = self._find_min_resolution(base_shape_projected)
        if minimum_resolution is not None and resolution <= minimum_resolution:
            self._suggest_minimum_resolution_which_still_fits(minimum_resolution)
            resolution = minimum_resolution - 1
        return int(resolution)

    def _suggest_minimum_resolution_which_still_fits(self, minimum_resolution):
        minimum_resolution_which_still_fits = str(minimum_resolution - 1)
        hexagon_edges = constants.H3_UTILS["average_hexagon_edge_length"]
        suggestion = hexagon_edges[minimum_resolution_which_still_fits] / 100
        warnings.warn(
            "The cell side-length you provided is too large to cover the input area. "
            f"Try something smaller, e.g. Side-Length {suggestion} Km",
            UserWarning,
            stacklevel=2,
        )

    def _handle_polyfill(self, base_shape, resolution):
        geometry, _ops = _require_shapely()
        geom = base_shape.geometry.iloc[0] if hasattr(base_shape, "geometry") else base_shape
        if isinstance(geom, geometry.MultiPolygon):
            hexagons = []
            for polygon in geom.geoms:
                hexagons.extend(self._get_hexagons(polygon, resolution))
            return list(dict.fromkeys(hexagons))
        return list(self._get_hexagons(geom, resolution))

    def _extract_geometry(self, base_shape):
        return base_shape.geometry.__geo_interface__["features"][0]["geometry"]

    def _get_hexagons(self, x, resolution):
        h3 = _require_h3()
        if hasattr(h3, "polyfill"):
            hexagons = h3.polyfill(x.__geo_interface__, resolution, geo_json_conformant=True)
        else:
            hexagons = h3.geo_to_cells(x.__geo_interface__, resolution)
        return list(hexagons)

    def _create_hexagon_polygons(self, hexagon_ids):
        gpd = _require_geopandas()
        geometry, _ops = _require_shapely()
        h3 = _require_h3()
        if hasattr(h3, "h3_to_geo_boundary"):
            def boundary(hexagon_id):
                return h3.h3_to_geo_boundary(hexagon_id, geo_json=True)
        else:
            def boundary(hexagon_id):
                return [(lng, lat) for lat, lng in h3.cell_to_boundary(hexagon_id)]
        return gpd.GeoDataFrame(
            {
                "geometry": [geometry.Polygon(boundary(hexagon_id)) for hexagon_id in hexagon_ids],
                "H3_INDEX": hexagon_ids,
            },
            crs=constants.DEFAULT_CRS,
        )

    def _add_tile_id(self, hexagon_polygons, tile_id_col: str = constants.TILE_ID):
        hexagon_polygons[tile_id_col] = hexagon_polygons.index
        hexagon_polygons[tile_id_col] = hexagon_polygons[tile_id_col].astype("str")
        return hexagon_polygons

    def _meters_to_resolution(self, meters):
        hexagon_side_length = self._meters_to_kilometers(meters)
        average_hexagon_edge_lengths = self._load_h3_utils("average_hexagon_edge_length")
        return int((np.abs(average_hexagon_edge_lengths - hexagon_side_length)).argmin())

    def _meters_to_kilometers(self, meters):
        return meters / 1000

    def _load_h3_utils(self, util):
        return np.asarray(list(constants.H3_UTILS[util].values()))

    def _find_min_resolution(self, base_shape):
        candidates = np.where(
            self._load_h3_utils("average_hexagon_area") > self._squared_meters_to_squared_kilometers(base_shape)
        )[0]
        if len(candidates) == 0:
            return None
        return int(candidates[-1])

    def _squared_meters_to_squared_kilometers(self, squared_meters):
        return float(squared_meters.area.values[0]) / 1000000


tiler.register_tiler("h3_tessellation", H3TessellationTiler())


__all__ = [
    "H3TessellationTiler",
    "SquaredTessellationTiler",
    "TessellationTiler",
    "TessellationTilers",
    "VoronoiTessellationTiler",
    "tiler",
]
