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

from .common import COLOR_ROLES, base_option, norm_width, resolve_palette, thin_points
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
    """Immutable, notebook-renderable chart specification.

    Returned by `ecdf`, `bar`, `scatter`, `heatmap`, `boxplot` and `map`; you do not build one directly. As the last expression of a notebook cell it draws itself. `to_dict` returns the ECharts option for a web application, and `to_html`, `to_json` and `to_svg` write portable files. `render` returns the lower-level `EChartsFigure`.

    Parameters
    ----------
    kind : str
        Chart type: ``"ecdf"``, ``"bar"``, ``"scatter"``, ``"heatmap"``, ``"boxplot"`` or ``"map"``.
    width : str
        CSS sizes.
    height : str
        CSS sizes.
    background : str
        Background colour taken from the palette.
    bundle_libs : bool
        Whether the JavaScript is embedded in the chart.
    display : {"html", "svg"}
        Notebook display mode.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.bar(pd.DataFrame({"who": ["a", "b"], "n": [3, 5]}), category_col="who", value_col="n")
    >>> sorted(chart.to_dict())[:3]
    ['_meta', 'animation', 'backgroundColor']
    """

    kind: str
    _option: dict[str, Any] = field(repr=False)
    width: str = "600px"
    height: str = "420px"
    background: str = "white"
    bundle_libs: bool = True
    display: str = "html"

    def render(self) -> EChartsFigure:
        """Return the lower-level `EChartsFigure` with the same option and settings."""
        return EChartsFigure(
            copy.deepcopy(self._option), width=self.width, height=self.height,
            background=self.background, bundle_libs=self.bundle_libs, display=self.display,
        )

    @property
    def option(self) -> dict[str, Any]:
        """A copy of the ECharts option; changing it does not change the chart."""
        return copy.deepcopy(self._option)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the ECharts option, ready to send to a web page.

        Returns
        -------
        dict
            The ECharts option, including a ``_meta`` entry with the chart type.
        """
        return self.render().to_dict()

    def to_json(self, filepath: str | Path) -> Path:
        """Write the ECharts option as a JSON file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        return self.render().to_json(filepath)

    def to_html(self, filepath: str | Path) -> Path:
        """Write a standalone HTML page that draws the chart.

        The page embeds the JavaScript libraries, so it opens without a network
        connection (map tiles excepted).

        Parameters
        ----------
        filepath : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        return self.render().to_html(filepath)

    def to_svg(self, filepath: str | Path | None = None) -> str | Path:
        """Draw the chart as a static SVG picture.

        Parameters
        ----------
        filepath : str or pathlib.Path, optional
            Destination file. When omitted the SVG text is returned.

        Returns
        -------
        str or pathlib.Path
            The SVG text, or the file written.

        Raises
        ------
        ValueError
            For maps and STVD comparisons, which can only be drawn with JavaScript.
        """
        return self.render().to_svg(filepath)

    def _repr_html_(self) -> str | None:
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
    """Draw the empirical cumulative distribution of a numeric column, optionally against a second result.

    The column may hold one number per row or one list of numbers per row (as `jump_lengths` returns); lists are flattened. Missing values are ignored. The curve stops at ``cdf_cutoff`` so that a long tail does not stretch the axis.

    Parameters
    ----------
    data : dataframe
        Any pandas, Polars or other Narwhals-compatible eager dataframe.
    value_col : str
        Column with the values, scalar or list-valued.
    second : dataframe, optional
        A second result with the same column, drawn as a dashed series for comparison.
    labels : sequence of two str, optional
        Series names for the first and second result. Default ``("first", "second")``.
    x_label : str, optional
        X-axis label; defaults to ``value_col``.
    x_unit : str, optional
        Unit appended to the tooltip values. Default empty.
    cdf_cutoff : float, optional
        Cumulative probability in (0, 1] at which each curve stops. Default 0.98.
    title : str, optional
        Chart title. Default ``"Empirical cumulative distribution"``.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If ``value_col`` is missing or has no finite numeric value, ``cdf_cutoff`` is outside (0, 1], or ``labels`` does not have two entries when ``second`` is given.
    TypeError
        If ``data`` is not an eager dataframe.

    See Also
    --------
    bar : Categorical bars.
    plot_jump_lengths_ecdf : Two-array ECDF with jump-length defaults.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.ecdf(pd.DataFrame({"jump_km": [0.8, 1.2, 2.4]}), value_col="jump_km", x_unit="km")
    >>> chart.to_dict()["series"][0]["type"]
    'line'
    """
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
            {"name": name, "type": "line", "data": thin_points(_ecdf(values, cdf_cutoff)), "showSymbol": False,
             "smooth": 0.35, "lineStyle": {"color": color, "width": 2.4, **({"type": dash} if dash else {})}}
            for name, values, color, dash in series_inputs
        ],
    })
    return _chart(option, "ecdf", palette, width, height, bundle_libs, display)


def bar(data: Any, *, category_col: str, value_col: str, series_col: str | None = None,
        title: str = "", palette: str | dict = "warm", width: int | str = 600, height: str = "420px",
        bundle_libs: bool = True, display: str = "html") -> Chart:
    """Draw a categorical bar chart.

    One bar per category; when ``series_col`` is given the bars are grouped, one series per value of that column. A category/series pair with no row is drawn as 0, and when it has several rows only the first is used, so aggregate first.

    Parameters
    ----------
    data : dataframe
        Any pandas, Polars or other Narwhals-compatible eager dataframe.
    category_col : str
        Column with the category on the x-axis.
    value_col : str
        Numeric column with the bar height.
    series_col : str, optional
        Column that splits the bars into groups. Default none: a single series named after ``value_col``.
    title : str, optional
        Chart title. Default empty.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If a named column is missing.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.bar(pd.DataFrame({"who": ["a", "b"], "n": [3, 5]}), category_col="who", value_col="n")
    >>> chart.to_dict()["xAxis"]["data"]
    ['a', 'b']
    """
    rows = _records(data, [category_col, value_col] + ([series_col] if series_col else []))
    p = resolve_palette(palette)
    categories = list(dict.fromkeys(str(row[category_col]) for row in rows))
    groups = list(dict.fromkeys(str(row[series_col]) for row in rows)) if series_col else [value_col]
    option = base_option(title, p, "category_bar")  # not "bar": that type formats values as percentages
    option.update({"legend": {"data": groups}, "tooltip": {"trigger": "axis"},
                   "xAxis": {"type": "category", "data": categories}, "yAxis": {"type": "value"},
                   "series": [{"name": group, "type": "bar", "data": [next((float(row[value_col]) for row in rows if str(row[category_col]) == category and (not series_col or str(row[series_col]) == group)), 0.0) for category in categories], "itemStyle": {"color": p[COLOR_ROLES[index % len(COLOR_ROLES)]]}} for index, group in enumerate(groups)]})
    return _chart(option, "bar", palette, width, height, bundle_libs, display)


def scatter(data: Any, *, x_col: str, y_col: str, series_col: str | None = None, title: str = "",
            palette: str | dict = "warm", width: int | str = 600, height: str = "420px",
            bundle_libs: bool = True, display: str = "html") -> Chart:
    """Draw a scatter chart of two numeric columns.

    Use ``series_col`` to overlay several groups, for example observed points and a fitted curve.

    Parameters
    ----------
    data : dataframe
        Any pandas, Polars or other Narwhals-compatible eager dataframe.
    x_col : str
        Numeric column on the x-axis.
    y_col : str
        Numeric column on the y-axis.
    series_col : str, optional
        Column that splits the points into coloured series. Default none.
    title : str, optional
        Chart title. Default empty.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If a named column is missing.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.scatter(pd.DataFrame({"x": [1, 2], "y": [2, 4]}), x_col="x", y_col="y")
    >>> chart.to_dict()["series"][0]["data"]
    [[1.0, 2.0], [2.0, 4.0]]
    """
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
    """Draw a heatmap from a long-form table of x, y and value columns.

    One cell per row; the colour scale runs from 0 to the largest value. A pair with no row is left empty.

    Parameters
    ----------
    data : dataframe
        Any pandas, Polars or other Narwhals-compatible eager dataframe.
    x_col : str
        Column with the horizontal category.
    y_col : str
        Column with the vertical category.
    value_col : str
        Numeric column that sets the colour.
    title : str, optional
        Chart title. Default empty.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If a named column is missing.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.heatmap(pd.DataFrame({"h": ["8", "9"], "who": ["a", "a"], "n": [1, 3]}), x_col="h", y_col="who", value_col="n")
    >>> chart.to_dict()["series"][0]["data"]
    [[0, 0, 1.0], [1, 0, 3.0]]
    """
    rows = _records(data, [x_col, y_col, value_col])
    p = resolve_palette(palette)
    xs, ys = list(dict.fromkeys(str(row[x_col]) for row in rows)), list(dict.fromkeys(str(row[y_col]) for row in rows))
    option = base_option(title, p, "heatmap")
    option.update({"tooltip": {"position": "top"}, "xAxis": {"type": "category", "data": xs}, "yAxis": {"type": "category", "data": ys}, "visualMap": {"min": 0, "max": max(float(row[value_col]) for row in rows), "calculable": True}, "series": [{"type": "heatmap", "data": [[xs.index(str(row[x_col])), ys.index(str(row[y_col])), float(row[value_col])] for row in rows]}]})
    return _chart(option, "heatmap", palette, width, height, bundle_libs, display)


def boxplot(data: Any, *, category_col: str, value_col: str, series_col: str | None = None, title: str = "",
            palette: str | dict = "warm", width: int | str = 600, height: str = "420px", bundle_libs: bool = True,
            display: str = "html") -> Chart:
    """Draw a box plot of a numeric column for each category.

    Each box spans the quartiles with the median inside and whiskers at the minimum and maximum. With ``series_col`` the boxes are grouped; a group with no rows is skipped.

    Parameters
    ----------
    data : dataframe
        Any pandas, Polars or other Narwhals-compatible eager dataframe.
    category_col : str
        Column with the category on the x-axis.
    value_col : str
        Numeric column summarised by each box.
    series_col : str, optional
        Column that splits each category into groups. Default none.
    title : str, optional
        Chart title. Default empty.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If a named column is missing.

    Examples
    --------
    >>> from fastmob import vis
    >>> import pandas as pd
    >>> chart = vis.boxplot(pd.DataFrame({"who": ["a"] * 4, "d": [1, 2, 3, 4]}), category_col="who", value_col="d")
    >>> chart.to_dict()["series"][0]["data"]
    [[1.0, 1.75, 2.5, 3.25, 4.0]]
    """
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
    """Draw polygons from a GeoJSON FeatureCollection as a map chart.

    No GIS library is needed. Maps are interactive only: ``to_svg`` and ``display="svg"`` raise ``ValueError`` for this chart.

    Parameters
    ----------
    geojson : dict
        A GeoJSON object whose ``type`` is ``"FeatureCollection"``.
    title : str, optional
        Chart title. Default ``"Map"``.
    palette : str or dict, optional
        Name of a built-in palette (``"warm"``, ``"ink"``, ``"terracotta"``, ``"forest"``) or a mapping with the same keys as `PALETTES`. Default ``"warm"``.
    width : int or str, optional
        Chart width: an integer in pixels or a CSS length such as ``"100%"``.
    height : str, optional
        Chart height as a CSS length such as ``"420px"``.
    bundle_libs : bool, optional
        Embed the ECharts (and, for maps, Leaflet) JavaScript in the chart. Pass False when the page already loads it, for example with `get_resource_bundle`. Default True.
    display : {"html", "svg"}, optional
        How a notebook shows the chart: ``"html"`` is interactive, ``"svg"`` is a static picture that also renders where JavaScript is not run, such as this documentation. Default ``"html"``.

    Returns
    -------
    Chart
        An immutable chart specification; it draws itself in a notebook and can be exported.

    Raises
    ------
    ValueError
        If ``geojson`` is not a FeatureCollection.
    """
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson must be a FeatureCollection")
    p = resolve_palette(palette)
    option = base_option(title, p, "map")
    option.update({"geoJSON": copy.deepcopy(geojson), "series": []})
    return _chart(option, "map", palette, width, height, bundle_libs, display)


__all__ = ["Chart", "bar", "boxplot", "ecdf", "heatmap", "map", "scatter"]
