from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .brand import FONT_SANS, FONT_SERIF, PALETTES
from .common import norm_width
from .figure import EChartsFigure

DEFAULT_TILE_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
DEFAULT_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)
STVD_COLORS = [
    ["#91bfdb", "#f7f7f7", "#f4a582"],
    ["#4393c3", "#bdbdbd", "#d6604d"],
    ["#2166ac", "#6e6e6e", "#b2182b"],
]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _resolution(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise ValueError(f"{label} must be an integer in [0, 15]")
    return value


def _coordinate_pairs(coordinates: Any, geometry_type: str, label: str) -> list[tuple[float, float]]:
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError(f"{label} must contain non-empty coordinate arrays")

    pairs: list[tuple[float, float]] = []

    def visit(value: Any, depth: int) -> None:
        if depth == 0:
            if not isinstance(value, list) or len(value) < 2:
                raise ValueError(f"{label} contains an invalid coordinate")
            longitude = _finite_number(value[0], f"{label} longitude")
            latitude = _finite_number(value[1], f"{label} latitude")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(f"{label} coordinates must use WGS84 longitude/latitude")
            pairs.append((longitude, latitude))
            return
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} contains an invalid coordinate array")
        for child in value:
            visit(child, depth - 1)

    visit(coordinates, 2 if geometry_type == "Polygon" else 3)
    return pairs


def _classify(volume_diff: float, peak_shift: float, threshold: float) -> tuple[int, int, int]:
    if volume_diff < -threshold:
        x_bin = 0
    elif volume_diff <= threshold:
        x_bin = 1
    else:
        x_bin = 2

    if peak_shift <= 2:
        y_bin = 0
    elif peak_shift <= 5:
        y_bin = 1
    else:
        y_bin = 2
    return x_bin, y_bin, y_bin * 3 + x_bin


def _prepare_layers(
    layers: Mapping[int, dict[str, Any]],
    *,
    area_property: str,
    volume_diff_property: str,
    peak_shift_property: str,
    volume_threshold: float,
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("layers must be a non-empty mapping of resolution to GeoJSON")

    prepared: dict[str, dict[str, Any]] = {}
    all_pairs: list[tuple[float, float]] = []
    for raw_resolution, raw_layer in layers.items():
        resolution = _resolution(raw_resolution, "layer resolution")
        if not isinstance(raw_layer, dict) or raw_layer.get("type") != "FeatureCollection":
            raise ValueError(f"layer {resolution} must be a GeoJSON FeatureCollection")
        features = raw_layer.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"layer {resolution} must contain at least one feature")

        layer = copy.deepcopy(raw_layer)
        for index, feature in enumerate(layer["features"]):
            label = f"layer {resolution} feature {index}"
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise ValueError(f"{label} must be a GeoJSON Feature")
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict) or geometry.get("type") not in {
                "Polygon",
                "MultiPolygon",
            }:
                raise ValueError(f"{label} geometry must be Polygon or MultiPolygon")
            all_pairs.extend(
                _coordinate_pairs(geometry.get("coordinates"), geometry["type"], f"{label} geometry")
            )

            properties = feature.get("properties")
            if not isinstance(properties, dict):
                raise TypeError(f"{label} must contain a properties object")
            for property_name in (area_property, volume_diff_property, peak_shift_property):
                if property_name not in properties:
                    raise ValueError(f"{label} is missing property {property_name!r}")

            volume_diff = _finite_number(
                properties[volume_diff_property], f"{label} property {volume_diff_property!r}"
            )
            peak_shift = _finite_number(
                properties[peak_shift_property], f"{label} property {peak_shift_property!r}"
            )
            if not 0 <= peak_shift <= 12:
                raise ValueError(f"{label} peak shift must be in [0, 12] hours")

            x_bin, y_bin, bivariate_class = _classify(
                volume_diff, peak_shift, volume_threshold
            )
            properties["_fastmobVis"] = {
                "area": str(properties[area_property]),
                "volumeDiff": volume_diff,
                "peakShift": peak_shift,
                "volumeBin": x_bin,
                "peakBin": y_bin,
                "class": bivariate_class,
                "color": STVD_COLORS[y_bin][x_bin],
            }
        prepared[str(resolution)] = layer

    west = min(pair[0] for pair in all_pairs)
    east = max(pair[0] for pair in all_pairs)
    south = min(pair[1] for pair in all_pairs)
    north = max(pair[1] for pair in all_pairs)
    return prepared, [(west + east) / 2, (south + north) / 2]


def _prepare_zoom_mapping(
    zoom_to_resolution: Mapping[int, int], available_resolutions: set[int]
) -> list[list[int]]:
    if not isinstance(zoom_to_resolution, Mapping) or not zoom_to_resolution:
        raise ValueError("zoom_to_resolution must be a non-empty mapping")

    mapping: list[list[int]] = []
    for raw_zoom, raw_resolution in zoom_to_resolution.items():
        if isinstance(raw_zoom, bool) or not isinstance(raw_zoom, int) or raw_zoom < 0:
            raise ValueError("zoom thresholds must be non-negative integers")
        resolution = _resolution(raw_resolution, "zoom resolution")
        if resolution not in available_resolutions:
            raise ValueError(
                f"zoom_to_resolution references missing layer resolution {resolution}"
            )
        mapping.append([raw_zoom, resolution])

    mapping.sort(key=lambda item: item[0])
    if mapping[0][0] != 0:
        raise ValueError("zoom_to_resolution must define a resolution for zoom 0")
    return mapping


def _resolution_for_zoom(mapping: Sequence[Sequence[int]], zoom: int) -> int:
    resolution = mapping[0][1]
    for threshold, candidate in mapping:
        if zoom < threshold:
            break
        resolution = candidate
    return resolution


def plot_stvd_comparison(
    layers: Mapping[int, dict[str, Any]],
    *,
    area_property: str = "area",
    volume_diff_property: str = "volume_diff_pct",
    peak_shift_property: str = "peak_shift_hours",
    volume_threshold: float = 3.0,
    zoom_to_resolution: Mapping[int, int] | None = None,
    center: Sequence[float] | None = None,
    zoom: int = 10,
    tile_url: str = DEFAULT_TILE_URL,
    attribution: str = DEFAULT_ATTRIBUTION,
    title: str = "STVD comparison",
    width: int | str = "100%",
    height: int | str = "600px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Plot a zoom-adaptive bivariate STVD comparison over a Leaflet basemap."""
    threshold = _finite_number(volume_threshold, "volume_threshold")
    if threshold <= 0:
        raise ValueError("volume_threshold must be greater than zero")
    if isinstance(zoom, bool) or not isinstance(zoom, int) or zoom < 0:
        raise ValueError("zoom must be a non-negative integer")
    if not all(
        isinstance(value, str) and value
        for value in (area_property, volume_diff_property, peak_shift_property)
    ):
        raise ValueError("property names must be non-empty strings")
    if not isinstance(tile_url, str) or not tile_url:
        raise ValueError("tile_url must be a non-empty string")
    if not isinstance(attribution, str):
        raise TypeError("attribution must be a string")

    prepared_layers, computed_center = _prepare_layers(
        layers,
        area_property=area_property,
        volume_diff_property=volume_diff_property,
        peak_shift_property=peak_shift_property,
        volume_threshold=threshold,
    )
    available_resolutions = {int(value) for value in prepared_layers}
    if zoom_to_resolution is None:
        # Match each H3 resolution to the Leaflet zoom at which its cells read well
        # (~res + 3, the common H3<->web-map-zoom heuristic) so zooming steps through
        # coarse -> fine smoothly instead of snapping to the finest layer. The coarsest
        # layer anchors zoom 0 so every zoom level has a resolution.
        ordered = sorted(available_resolutions)
        zoom_to_resolution = {0: ordered[0]}
        for resolution in ordered[1:]:
            zoom_to_resolution[resolution + 3] = resolution
    zoom_mapping = _prepare_zoom_mapping(zoom_to_resolution, available_resolutions)

    if center is None:
        map_center = computed_center
    else:
        if isinstance(center, (str, bytes)) or not isinstance(center, Sequence) or len(center) != 2:
            raise ValueError("center must be a (longitude, latitude) pair")
        longitude = _finite_number(center[0], "center longitude")
        latitude = _finite_number(center[1], "center latitude")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("center must use WGS84 longitude/latitude")
        map_center = [longitude, latitude]

    initial_resolution = _resolution_for_zoom(zoom_mapping, zoom)
    initial_feature_count = len(prepared_layers[str(initial_resolution)]["features"])
    palette = PALETTES["warm"]
    option = {
        "animation": False,
        # Transparent so the Leaflet basemap tiles drawn beneath the ECharts canvas
        # remain visible; an opaque fill would paint over the map.
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": FONT_SANS, "color": palette["axis"]},
        "_meta": {
            "chartType": "stvd_comparison",
            "layers": prepared_layers,
            "zoomToResolution": zoom_mapping,
            "initialResolution": initial_resolution,
            "volumeThreshold": threshold,
            "colors": STVD_COLORS,
        },
        "title": {
            "text": title,
            "left": 16,
            "top": 12,
            "padding": [7, 10],
            "backgroundColor": "rgba(251,248,241,0.92)",
            "textStyle": {
                "fontFamily": FONT_SERIF,
                "fontWeight": 500,
                "fontSize": 22,
                "color": palette["axis"],
            },
        },
        "tooltip": {
            "trigger": "item",
            "confine": True,
            "backgroundColor": palette["bg"],
            "borderColor": palette["axis"],
            "borderWidth": 1.25,
            "textStyle": {"fontFamily": FONT_SANS, "color": palette["axis"]},
        },
        "leaflet": {
            "center": map_center,
            "zoom": zoom,
            "roam": True,
            "leafletOption": {"zoomControl": True},
            "tiles": [
                {
                    "label": "basemap",
                    "urlTemplate": tile_url,
                    "options": {"attribution": attribution, "maxZoom": 20},
                }
            ],
        },
        "series": [
            {
                "name": "STVD comparison",
                "type": "custom",
                "coordinateSystem": "leaflet",
                "data": [[index] for index in range(initial_feature_count)],
                "silent": False,
                "zlevel": 2,
            }
        ],
    }
    return EChartsFigure(
        option,
        width=norm_width(width),
        height=norm_width(height),
        background=palette["bg"],
        bundle_libs=bundle_libs,
        display=display,
    )


__all__ = ["plot_stvd_comparison"]
