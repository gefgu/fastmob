from __future__ import annotations

import copy
import html
import importlib.resources
import json
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

from ._core import render_option_svg
from .brand import FONT_URL

_FORMATTER_RESOURCES: dict[str, tuple[str, ...]] = {
    "ecdf": ("formatter_ecdf.js",),
    "bar": ("formatter_percent.js", "formatter_bar.js"),
    "visit_purpose_comparison": (
        "formatter_percent.js",
        "formatter_visit_purpose_comparison.js",
    ),
    "transition": ("formatter_percent.js", "formatter_heatmap.js"),
    "daily_activity": ("formatter_percent.js", "formatter_heatmap.js"),
    "transition_difference": ("formatter_difference_heatmap.js",),
    "daily_activity_difference": ("formatter_difference_heatmap.js",),
    "motif_literature_comparison": (
        "formatter_percent.js",
        "formatter_motif.js",
    ),
    "mobility_law": ("formatter_mobility_law.js",),
    "stvd_comparison": ("formatter_stvd_comparison.js",),
}

_BROWSER_JS_RESOURCES: dict[str, tuple[str, ...]] = {
    "stvd_comparison": ("leaflet.js", "echarts-extension-leaflet.js"),
}

_CSS_RESOURCES: dict[str, tuple[str, ...]] = {
    "stvd_comparison": ("leaflet.css", "stvd_comparison.css"),
}

# Resources provided by the shared bundle (get_resource_bundle) — skip per-chart when bundle_libs=False.
_BUNDLED_RESOURCES: frozenset[str] = frozenset(
    {"echarts.min.js", "leaflet.js", "echarts-extension-leaflet.js", "leaflet.css"}
)

_DISPLAY_MODES = {"html", "svg"}
_UNSUPPORTED_SVG_CHART_TYPES = {"stvd_comparison"}

_ECHARTS_HTML_TEMPLATE = Template(
    """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="$font_url">
  <style>$chart_css</style>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; background: $background; }
    #$element_id { width: 100%; height: 100%; min-height: $height; }
  </style>
</head>
<body>
  <div id="$element_id"></div>
<script>$echarts_js</script>
<script>$browser_js</script>
<script>
(function() {
  const option = JSON.parse($option_literal);
  const meta = option._meta || {};
  delete option._meta;
$formatter_js  const chart = echarts.init(document.getElementById("$element_id"));
  chart.setOption(option);
  if (typeof fastmobVisInitialize === "function") {
    fastmobVisInitialize(chart, document.getElementById("$element_id"));
  }
  window.addEventListener("resize", function() { chart.resize(); });
})();
</script>
</body>
</html>
"""
)

_INLINE_TEMPLATE = Template(
    """<div class="fastmob-vis-widget" style="flex: 1 1 $width; min-width: 0; overflow: hidden;">
<style>$chart_css</style>
<div id="$element_id" style="width: 100%; height: $height;"></div>
<script>
(function() {
  const option = JSON.parse($option_literal);
  const meta = option._meta || {};
  delete option._meta;
$formatter_js  const chart = echarts.init(document.getElementById("$element_id"));
  chart.setOption(option);
  if (typeof fastmobVisInitialize === "function") {
    fastmobVisInitialize(chart, document.getElementById("$element_id"));
  }
  window.addEventListener("resize", function() { chart.resize(); });
})();
</script>
</div>
"""
)


@lru_cache(maxsize=None)
def _static_js(filename: str) -> str:
    with importlib.resources.open_text("fastmob_vis.static", filename, encoding="utf-8") as resource:
        return resource.read()


def _formatter_js(chart_type: str | None) -> str:
    resource_names = _FORMATTER_RESOURCES.get(chart_type or "ecdf", ())
    return "".join(_static_js(resource_name) for resource_name in resource_names)


def _browser_js(chart_type: str | None, *, bundle_libs: bool = True) -> str:
    resource_names = _BROWSER_JS_RESOURCES.get(chart_type or "", ())
    if not bundle_libs:
        resource_names = tuple(r for r in resource_names if r not in _BUNDLED_RESOURCES)
    return "\n".join(_static_js(resource_name) for resource_name in resource_names)


def _chart_css(chart_type: str | None, *, bundle_libs: bool = True) -> str:
    resource_names = _CSS_RESOURCES.get(chart_type or "", ())
    if not bundle_libs:
        resource_names = tuple(r for r in resource_names if r not in _BUNDLED_RESOURCES)
    return "\n".join(_static_js(resource_name) for resource_name in resource_names)


def get_resource_bundle(*, echarts: bool = True, leaflet: bool = False) -> str:
    """Return inline HTML (script/style tags) to inject once in a parent document's <head>.

    Use with bundle_libs=False on all plot functions to share heavy libraries across
    multiple charts in a single HTML document rather than embedding them per chart.
    """
    parts: list[str] = []
    if echarts:
        parts.append(f"<script>{_static_js('echarts.min.js')}</script>")
    if leaflet:
        parts.append(f"<style>{_static_js('leaflet.css')}</style>")
        parts.append(f"<script>{_static_js('leaflet.js')}</script>")
        parts.append(f"<script>{_static_js('echarts-extension-leaflet.js')}</script>")
    return "\n".join(parts)


@dataclass(frozen=True, init=False)
class EChartsFigure:
    """Notebook-renderable ECharts option wrapper."""

    _option: dict[str, Any] = field(repr=False)
    width: str = "600px"
    height: str = "420px"
    background: str = "white"
    bundle_libs: bool = True
    display: str = "html"

    def __init__(
        self,
        option: dict[str, Any],
        width: str = "600px",
        height: str = "420px",
        background: str = "white",
        bundle_libs: bool = True,
        display: str = "html",
    ) -> None:
        if not isinstance(option, dict):
            raise TypeError("option must be a dictionary")
        if display not in _DISPLAY_MODES:
            raise ValueError(f"display must be one of {sorted(_DISPLAY_MODES)}, got {display!r}")
        object.__setattr__(self, "_option", copy.deepcopy(option))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "bundle_libs", bundle_libs)
        object.__setattr__(self, "display", display)

    @property
    def option(self) -> dict[str, Any]:
        return copy.deepcopy(self._option)

    @property
    def option_json(self) -> str:
        return json.dumps(self._option)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._option)

    def to_json(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.write_text(self.option_json, encoding="utf-8")
        return path

    def to_html(self, filepath: str | Path) -> Path:
        """Write a standalone HTML document that renders the chart."""
        path = Path(filepath)
        path.write_text(self._to_html_string(), encoding="utf-8")
        return path

    def to_svg(self, filepath: str | Path | None = None) -> str | Path:
        """Return or write a static SVG rendering of the chart."""
        chart_type = self._option.get("_meta", {}).get("chartType")
        if chart_type in _UNSUPPORTED_SVG_CHART_TYPES:
            raise ValueError(f"SVG rendering is not supported for chart type {chart_type!r}")
        svg = render_option_svg(
            self.option_json,
            _pixel_size(self.width, "width"),
            _pixel_size(self.height, "height"),
        )
        if filepath is None:
            return svg
        path = Path(filepath)
        path.write_text(svg, encoding="utf-8")
        return path

    def _to_html_string(self) -> str:
        """Return a fully standalone HTML document (always bundles all libs)."""
        element_id = f"fastmob-vis-{uuid.uuid4().hex}"
        option_literal = json.dumps(self.option_json)
        height = html.escape(self.height, quote=True)
        background = html.escape(self.background, quote=True)
        font_url = html.escape(FONT_URL, quote=True)
        chart_type = self._option.get("_meta", {}).get("chartType")
        return _ECHARTS_HTML_TEMPLATE.substitute(
            element_id=element_id,
            font_url=font_url,
            background=background,
            height=height,
            echarts_js=_static_js("echarts.min.js"),
            browser_js=_browser_js(chart_type, bundle_libs=True),
            chart_css=_chart_css(chart_type, bundle_libs=True),
            option_literal=option_literal,
            formatter_js=_formatter_js(chart_type),
        )

    def _to_inline_snippet(self) -> str:
        """Return a div+script snippet that relies on echarts/leaflet from the parent document."""
        element_id = f"fastmob-vis-{uuid.uuid4().hex}"
        option_literal = json.dumps(self.option_json)
        chart_type = self._option.get("_meta", {}).get("chartType")
        return _INLINE_TEMPLATE.substitute(
            element_id=element_id,
            width=html.escape(self.width, quote=True),
            height=html.escape(self.height, quote=True),
            chart_css=_chart_css(chart_type, bundle_libs=False),
            formatter_js=_formatter_js(chart_type),
            option_literal=option_literal,
        )

    def _repr_html_(self) -> str:
        if not self.bundle_libs:
            return self._to_inline_snippet()
        iframe_srcdoc = html.escape(self._to_html_string(), quote=True)
        width = html.escape(self.width, quote=True)
        height = html.escape(self.height, quote=True)
        return (
            f'<iframe srcdoc="{iframe_srcdoc}" '
            f'style="width: {width}; height: {height}; border: 0;" '
            'sandbox="allow-scripts allow-same-origin"></iframe>'
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        if self.display == "svg":
            return {"image/svg+xml": self.to_svg()}
        return {"text/html": self._repr_html_()}


def _pixel_size(value: str, name: str) -> int:
    if value.endswith("px"):
        value = value[:-2]
    try:
        pixels = int(value)
    except ValueError as exc:
        raise ValueError(f"SVG rendering requires a fixed pixel {name}, got {value!r}") from exc
    if pixels <= 0:
        raise ValueError(f"SVG rendering requires a positive pixel {name}, got {pixels}")
    return pixels
