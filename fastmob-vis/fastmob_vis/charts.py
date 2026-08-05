"""Backend-neutral chart specifications for :mod:`fastmob.vis`.

The public API is deliberately chart-shaped rather than measure-shaped: callers
pass a Fastmob result dataframe plus the fields that encode it.  This keeps the
visualization package independent from individual measure return contracts while
still accepting every eager dataframe backend supported by Narwhals.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import narwhals as nw
import numpy as np

from .common import COLOR_ROLES, base_option, norm_width, resolve_palette
from .figure import EChartsFigure


def _frame(data: Any) -> nw.DataFrame:
    try:
        return nw.from_native(data, eager_only=True)
    except Exception as exc:
        raise TypeError("data must be a Narwhals-compatible eager dataframe") from exc


def _values(data: Any, column: str, *, flatten: bool = False) -> list[float]:
    frame = _frame(data)
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is not present in the data")
    values: list[Any] = frame.get_column(column).to_list()
    if flatten:
        values = [item for value in values for item in (value if isinstance(value, (list, tuple, np.ndarray)) else [value])]
    try:
        result = [float(value) for value in values if value is not None]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must contain numeric values") from exc
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"column {column!r} must contain at least one finite numeric value")
    return result


def _records(data: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    frame = _frame(data)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing chart columns: {missing}")
    selected = frame.select(list(columns)).to_dict(as_series=False)
    return [dict(zip(columns, values)) for values in zip(*(selected[column] for column in columns))]


def _ecdf(values: list[float], cutoff: float) -> list[list[float]]:
    if not 0 < cutoff <= 1:
        raise ValueError("cdf_cutoff must be in (0, 1]")
    ordered = sorted(values)
    result: list[list[float]] = []
    index = 0
    while index < len(ordered):
        value = ordered[index]
        end = index + 1
        while end < len(ordered) and ordered[end] == value:
            end += 1
        probability = end / len(ordered)
        result.append([value, min(probability, cutoff)])
        if probability >= cutoff:
            break
        index = end
    return result


@dataclass(frozen=True)
class Chart:
    """Immutable, notebook-renderable visualization specification."""

    kind: str
    _option: dict[str, Any] = field(repr=False)
    width: str = "600px"
    height: str = "420px"
    background: str = "white"
    bundle_libs: bool = True
    display: str = "html"

    def render(self) -> EChartsFigure:
        return EChartsFigure(
            copy.deepcopy(self._option), width=self.width, height=self.height,
            background=self.background, bundle_libs=self.bundle_libs, display=self.display,
        )

    @property
    def option(self) -> dict[str, Any]:
        return copy.deepcopy(self._option)

    def to_dict(self) -> dict[str, Any]:
        return self.render().to_dict()

    def to_json(self, filepath: str | Path) -> Path:
        return self.render().to_json(filepath)

    def to_html(self, filepath: str | Path) -> Path:
        return self.render().to_html(filepath)

    def to_svg(self, filepath: str | Path | None = None) -> str | Path:
        return self.render().to_svg(filepath)

    def _repr_html_(self) -> str:
        return self.render()._repr_html_()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return self.render()._repr_mimebundle_(include, exclude)


def _chart(option: dict[str, Any], kind: str, palette: str | dict, width: int | str, height: str,
           bundle_libs: bool, display: str) -> Chart:
    colors = resolve_palette(palette)
    return Chart(kind, option, norm_width(width), height, colors["bg"], bundle_libs, display)


def ecdf(
    data: Any,
    *,
    value_col: str,
    second: Any | None = None,
    labels: Sequence[str] = ("first", "second"),
    title: str = "Empirical cumulative distribution",
    x_label: str | None = None,
    x_unit: str = "",
    cdf_cutoff: float = 0.98,
    palette: str | dict = "warm",
    width: int | str = 600,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> Chart:
    """Create a one- or two-result ECDF from a numeric (including list-valued) column."""
    if second is not None and len(labels) != 2:
        raise ValueError("labels must contain two values when second is supplied")
    p = resolve_palette(palette)
    series_inputs = [(str(labels[0]), _values(data, value_col, flatten=True), p["observed"], None)]
    if second is not None:
        series_inputs.append((str(labels[1]), _values(second, value_col, flatten=True), p["synthetic"], [6, 4]))
    option = base_option(title, p, "ecdf")
    option["_meta"].update({"xLabel": x_label or value_col, "xUnit": x_unit})
    option.update({
        "legend": {"data": [item[0] for item in series_inputs], "right": 0, "top": 12},
        "grid": {"left": 76, "right": 28, "top": 64, "bottom": 64},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "value", "name": (x_label or value_col).upper(), "nameLocation": "middle", "nameGap": 38},
        "yAxis": {"type": "value", "name": "F(X)", "min": 0, "max": 1, "nameLocation": "middle", "nameGap": 52},
        "series": [
            {"name": name, "type": "line", "data": _ecdf(values, cdf_cutoff), "showSymbol": False,
             "smooth": 0.35, "lineStyle": {"color": color, "width": 2.4, **({"type": dash} if dash else {})}}
            for name, values, color, dash in series_inputs
        ],
    })
    return _chart(option, "ecdf", palette, width, height, bundle_libs, display)


def bar(data: Any, *, category_col: str, value_col: str, series_col: str | None = None,
        title: str = "", palette: str | dict = "warm", width: int | str = 600, height: str = "420px",
        bundle_libs: bool = True, display: str = "html") -> Chart:
    """Create a categorical bar chart from explicit dataframe field mappings."""
    rows = _records(data, [category_col, value_col] + ([series_col] if series_col else []))
    p = resolve_palette(palette)
    categories = list(dict.fromkeys(str(row[category_col]) for row in rows))
    groups = list(dict.fromkeys(str(row[series_col]) for row in rows)) if series_col else [value_col]
    option = base_option(title, p, "bar")
    option.update({"legend": {"data": groups}, "tooltip": {"trigger": "axis"},
                   "xAxis": {"type": "category", "data": categories}, "yAxis": {"type": "value"},
                   "series": [{"name": group, "type": "bar", "data": [next((float(row[value_col]) for row in rows if str(row[category_col]) == category and (not series_col or str(row[series_col]) == group)), 0.0) for category in categories], "itemStyle": {"color": p[COLOR_ROLES[index % len(COLOR_ROLES)]]}} for index, group in enumerate(groups)]})
    return _chart(option, "bar", palette, width, height, bundle_libs, display)


def scatter(data: Any, *, x_col: str, y_col: str, series_col: str | None = None, title: str = "",
            palette: str | dict = "warm", width: int | str = 600, height: str = "420px",
            bundle_libs: bool = True, display: str = "html") -> Chart:
    """Create a scatter chart; use ``series_col`` to overlay Fastmob results or fits."""
    rows = _records(data, [x_col, y_col] + ([series_col] if series_col else []))
    p = resolve_palette(palette)
    groups = list(dict.fromkeys(str(row[series_col]) for row in rows)) if series_col else ["values"]
    option = base_option(title, p, "scatter")
    option.update({"legend": {"data": groups}, "tooltip": {"trigger": "item"},
                   "xAxis": {"type": "value", "name": x_col}, "yAxis": {"type": "value", "name": y_col},
                   "series": [{"name": group, "type": "scatter", "data": [[float(row[x_col]), float(row[y_col])] for row in rows if not series_col or str(row[series_col]) == group], "itemStyle": {"color": p[COLOR_ROLES[index % len(COLOR_ROLES)]]}} for index, group in enumerate(groups)]})
    return _chart(option, "scatter", palette, width, height, bundle_libs, display)


def heatmap(data: Any, *, x_col: str, y_col: str, value_col: str, title: str = "", palette: str | dict = "warm",
            width: int | str = 600, height: str = "520px", bundle_libs: bool = True, display: str = "html") -> Chart:
    """Create a long-form heatmap from explicit x, y, and value columns."""
    rows = _records(data, [x_col, y_col, value_col])
    p = resolve_palette(palette)
    xs, ys = list(dict.fromkeys(str(row[x_col]) for row in rows)), list(dict.fromkeys(str(row[y_col]) for row in rows))
    option = base_option(title, p, "heatmap")
    option.update({"tooltip": {"position": "top"}, "xAxis": {"type": "category", "data": xs}, "yAxis": {"type": "category", "data": ys}, "visualMap": {"min": 0, "max": max(float(row[value_col]) for row in rows), "calculable": True}, "series": [{"type": "heatmap", "data": [[xs.index(str(row[x_col])), ys.index(str(row[y_col])), float(row[value_col])] for row in rows]}]})
    return _chart(option, "heatmap", palette, width, height, bundle_libs, display)


def boxplot(data: Any, *, category_col: str, value_col: str, series_col: str | None = None, title: str = "",
            palette: str | dict = "warm", width: int | str = 600, height: str = "420px", bundle_libs: bool = True,
            display: str = "html") -> Chart:
    """Create a grouped box plot from long-form dataframe data."""
    rows = _records(data, [category_col, value_col] + ([series_col] if series_col else []))
    p = resolve_palette(palette)
    categories = list(dict.fromkeys(str(row[category_col]) for row in rows))
    groups = list(dict.fromkeys(str(row[series_col]) for row in rows)) if series_col else [value_col]
    def stats(values: list[float]) -> list[float]:
        return [float(np.min(values)), float(np.percentile(values, 25)), float(np.median(values)), float(np.percentile(values, 75)), float(np.max(values))]
    option = base_option(title, p, "boxplot")
    option.update({"legend": {"data": groups}, "xAxis": {"type": "category", "data": categories}, "yAxis": {"type": "value"}, "series": [{"name": group, "type": "boxplot", "data": [stats([float(row[value_col]) for row in rows if str(row[category_col]) == category and (not series_col or str(row[series_col]) == group)]) for category in categories], "itemStyle": {"borderColor": p[COLOR_ROLES[index % len(COLOR_ROLES)]]}} for index, group in enumerate(groups)]})
    return _chart(option, "boxplot", palette, width, height, bundle_libs, display)


def map(geojson: dict[str, Any], *, title: str = "Map", palette: str | dict = "warm", width: int | str = "100%",
        height: str = "600px", bundle_libs: bool = True, display: str = "html") -> Chart:
    """Create a map chart from a GeoJSON FeatureCollection without GIS dependencies."""
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson must be a FeatureCollection")
    p = resolve_palette(palette)
    option = base_option(title, p, "map")
    option.update({"geoJSON": copy.deepcopy(geojson), "series": []})
    return _chart(option, "map", palette, width, height, bundle_libs, display)


__all__ = ["Chart", "bar", "boxplot", "ecdf", "heatmap", "map", "scatter"]
