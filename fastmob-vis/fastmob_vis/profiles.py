from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from .brand import FONT_MONO, FONT_SANS, FONT_SERIF
from .common import COLOR_ROLES, base_option, norm_width, resolve_palette
from .data import column
from .figure import EChartsFigure

# Canonical mobility-profile ordering (Amichi et al. 2020): a stable legend / axis
# order, with any unrecognised profile appended after these.
_CANONICAL_PROFILE_ORDER = ("Routiner", "Regular", "Scouter", "All")

# Default profile -> palette-role mapping (on-brand, distinct per profile, mirroring
# the source red/green/blue semantics: Scouter warm-red, Routiner green, Regular blue).
_DEFAULT_PROFILE_ROLES = {
    "Scouter": "synthetic",
    "Regular": "syn3",
    "Routiner": "syn2",
    "All": "baseline",
}

# Symbol shapes used to distinguish datasets overlaid on the same scatter; color
# still encodes the mobility profile, shape encodes which dataset a point came from.
_DATASET_SYMBOLS = ("circle", "triangle", "diamond", "rect", "roundRect", "pin")


def _ordered_profiles(profiles: Sequence[str]) -> list[str]:
    seen = list(dict.fromkeys(str(profile) for profile in profiles))
    ordered = [profile for profile in _CANONICAL_PROFILE_ORDER if profile in seen]
    ordered += [profile for profile in seen if profile not in ordered]
    return ordered


def _resolve_profile_colors(
    profiles: Sequence[str], p: dict, profile_colors: dict | None
) -> dict[str, str]:
    if profile_colors is not None:
        missing = [profile for profile in profiles if profile not in profile_colors]
        if missing:
            raise ValueError(f"profile_colors is missing entries for: {missing}")
        return {profile: profile_colors[profile] for profile in profiles}
    colors: dict[str, str] = {}
    fallback = list(COLOR_ROLES)
    fallback_index = 0
    for profile in profiles:
        role = _DEFAULT_PROFILE_ROLES.get(profile)
        if role is None:
            role = fallback[fallback_index % len(fallback)]
            fallback_index += 1
        colors[profile] = p[role]
    return colors


def _finite_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _value_axis(name: str, p: dict, *, rotate_name: bool) -> dict[str, Any]:
    axis = {
        "type": "value",
        "name": name.upper(),
        "nameLocation": "middle",
        "nameGap": 64 if rotate_name else 42,
        "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
        "axisLabel": {"fontFamily": FONT_MONO, "fontSize": 26, "color": p["axis"]},
        "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
        "axisTick": {"lineStyle": {"color": p["axis"]}},
        "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
    }
    if rotate_name:
        axis["nameRotate"] = 90
    return axis


def _as_profile_datasets(profiles: Any) -> dict[str, Any]:
    """Normalise ``profiles`` into an ordered name -> dataset mapping.

    A single dataset (dataframe-like, or a dict carrying ``degree_of_return``) maps
    to a single unnamed entry; a mapping of dataset name -> dataset is passed through
    so several datasets can be overlaid on one scatter.
    """
    if isinstance(profiles, Mapping) and "degree_of_return" not in profiles:
        return {str(name): data for name, data in profiles.items()}
    return {"": profiles}


def plot_mobility_profiles(
    profiles: Any,
    *,
    profile_col: str = "agent_type",
    profile_colors: dict | None = None,
    title: str = "Mobility profiles",
    palette: str | dict = "warm",
    width: int | str = 520,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Scatter of intermittency vs degree of return, colored by mobility profile.

    ``profiles`` is a dataframe-like or dict with ``degree_of_return``,
    ``intermittency`` and a profile-label column (``profile_col``). Points are
    grouped into one scatter series per profile so the clustering is visible.

    To compare datasets, pass a mapping of dataset name -> dataset; all datasets are
    overlaid on the same axes with color encoding the profile and symbol shape
    encoding the dataset.
    """
    datasets = _as_profile_datasets(profiles)
    dataset_names = list(datasets)
    multi = len(dataset_names) > 1 or dataset_names[0] != ""

    parsed: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    all_labels: list[str] = []
    for name, data in datasets.items():
        degree_of_return = _finite_array(column(data, "degree_of_return"), "degree_of_return")
        intermittency = _finite_array(column(data, "intermittency"), "intermittency")
        labels = [str(value) for value in column(data, profile_col)]
        if not (len(degree_of_return) == len(intermittency) == len(labels)):
            raise ValueError("degree_of_return, intermittency and profile columns must have equal length")
        parsed[name] = (degree_of_return, intermittency, labels)
        all_labels.extend(labels)

    p = resolve_palette(palette)
    ordered = _ordered_profiles(all_labels)
    colors = _resolve_profile_colors(ordered, p, profile_colors)
    symbols = {
        name: _DATASET_SYMBOLS[index % len(_DATASET_SYMBOLS)]
        for index, name in enumerate(dataset_names)
    }

    option = base_option(title, p, "mobility_profiles")
    legend_names: list[str] = []
    series: list[dict[str, Any]] = []
    for name in dataset_names:
        degree_of_return, intermittency, labels = parsed[name]
        symbol = symbols[name]
        for profile in ordered:
            points = [
                [float(degree_of_return[i]), float(intermittency[i])]
                for i, label in enumerate(labels)
                if label == profile
            ]
            if not points:
                continue
            series_name = f"{profile} · {name}" if multi else profile
            legend_names.append(series_name)
            series.append(
                {
                    "name": series_name,
                    "type": "scatter",
                    "symbol": symbol,
                    "data": points,
                    "symbolSize": 11,
                    "itemStyle": {
                        "color": colors[profile],
                        "opacity": 0.82,
                        "borderColor": p["bg"],
                        "borderWidth": 0.6,
                    },
                    "emphasis": {"scale": 1.3},
                    "z": 3,
                }
            )

    option.update(
        {
            "grid": {"left": 78, "right": 28, "top": 96, "bottom": 70, "containLabel": False},
            "legend": {
                "data": legend_names,
                "top": 54,
                "left": 0,
                "right": 0,
                "itemGap": 18,
                "textStyle": {"fontFamily": FONT_SANS, "fontSize": 14, "color": p["axis"]},
            },
            "tooltip": {
                "trigger": "item",
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
            },
            "xAxis": _value_axis("degree of return", p, rotate_name=False),
            "yAxis": _value_axis("intermittency", p, rotate_name=True),
            "series": series,
        }
    )
    return EChartsFigure(
        option,
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def _box_stats(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return [
        float(np.min(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
        float(np.max(values)),
    ]


def _grid_layout(count: int) -> tuple[int, int, list[dict[str, str]], list[dict[str, Any]]]:
    ncols = 2 if count > 1 else 1
    nrows = math.ceil(count / ncols)
    left_margin, right_margin, h_gap = 7.0, 3.0, 9.0
    top_margin, bottom_margin, v_gap = 16.0, 6.0, 13.0
    cell_w = (100.0 - left_margin - right_margin - h_gap * (ncols - 1)) / ncols
    cell_h = (100.0 - top_margin - bottom_margin - v_gap * (nrows - 1)) / nrows
    grids: list[dict[str, str]] = []
    title_anchors: list[dict[str, Any]] = []
    for index in range(count):
        row, col = divmod(index, ncols)
        left = left_margin + col * (cell_w + h_gap)
        top = top_margin + row * (cell_h + v_gap)
        grids.append(
            {
                "left": f"{left:.2f}%",
                "top": f"{top:.2f}%",
                "width": f"{cell_w:.2f}%",
                "height": f"{cell_h:.2f}%",
            }
        )
        title_anchors.append({"left": f"{left + cell_w / 2:.2f}%", "top": f"{max(top - 5.0, 0.0):.2f}%"})
    return nrows, ncols, grids, title_anchors


def plot_profile_metrics(
    datasets: Mapping[str, Any],
    *,
    metrics: Sequence[str] = ("regularity", "diversity", "stationarity", "entropy"),
    profile_col: str = "agent_type",
    profile_order: Sequence[str] = ("Scouter", "Regular", "Routiner"),
    title: str = "Mobility profile metrics",
    palette: str | dict = "warm",
    width: int | str = 900,
    height: str = "640px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Grouped box plots of mobility metrics, one sub-plot per metric.

    ``datasets`` maps a dataset name to a dataframe-like / dict carrying a profile
    column and the metric columns. Each sub-plot shows the profiles on the x-axis
    with one box per dataset (dodged), so a one-entry mapping is single-dataset.
    """
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("datasets must be a non-empty mapping of dataset names to profile data")
    metrics = [str(metric) for metric in metrics]
    if not metrics:
        raise ValueError("metrics must contain at least one metric column")
    profile_order = [str(profile) for profile in profile_order]

    p = resolve_palette(palette)
    dataset_names = [str(name) for name in datasets]

    # values[metric][dataset][profile] -> ndarray of metric values
    values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for metric in metrics:
        values[metric] = {}
        for name, data in datasets.items():
            labels = [str(label) for label in column(data, profile_col)]
            # Metric columns may carry NaN for sparse users; _box_stats drops those.
            metric_values = np.asarray(column(data, metric), dtype=float).ravel()
            if len(labels) != len(metric_values):
                raise ValueError(f"dataset {name!r} profile and {metric!r} columns must have equal length")
            by_profile: dict[str, np.ndarray] = {}
            for profile in profile_order:
                mask = np.asarray([label == profile for label in labels], dtype=bool)
                by_profile[profile] = metric_values[mask]
            values[metric][str(name)] = by_profile

    _nrows, _ncols, grids, title_anchors = _grid_layout(len(metrics))

    titles: list[dict[str, Any]] = [
        {
            "text": title,
            "left": 0,
            "top": 0,
            "textStyle": {
                "fontFamily": FONT_SERIF,
                "fontWeight": 500,
                "fontSize": 32,
                "color": p["axis"],
            },
        }
    ]
    x_axes: list[dict[str, Any]] = []
    y_axes: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        titles.append(
            {
                "text": metric.replace("_", " ").title(),
                "left": title_anchors[index]["left"],
                "top": title_anchors[index]["top"],
                "textAlign": "center",
                "textStyle": {"fontFamily": FONT_SANS, "fontWeight": 500, "fontSize": 22, "color": p["axis"]},
            }
        )
        x_axes.append(
            {
                "gridIndex": index,
                "type": "category",
                "data": profile_order,
                "axisLabel": {"fontFamily": FONT_SANS, "fontSize": 26, "color": p["axis"], "interval": 0},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
                "splitArea": {"show": False},
            }
        )
        y_axes.append(
            {
                "gridIndex": index,
                "type": "value",
                "min": 0,
                "axisLabel": {"fontFamily": FONT_MONO, "fontSize": 24, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
            }
        )
        for dataset_index, name in enumerate(dataset_names):
            color = p[COLOR_ROLES[dataset_index % len(COLOR_ROLES)]]
            series.append(
                {
                    "name": name,
                    "type": "boxplot",
                    "xAxisIndex": index,
                    "yAxisIndex": index,
                    "data": [_box_stats(values[metric][name][profile]) for profile in profile_order],
                    "itemStyle": {"color": p["bg"], "borderColor": color, "borderWidth": 1.6},
                    "boxWidth": [10, 40],
                }
            )

    option = base_option(title, p, "profile_metrics")
    option["title"] = titles
    option.update(
        {
            "grid": grids,
            "legend": {
                "data": dataset_names,
                "top": 8,
                "right": 0,
                "itemWidth": 18,
                "itemHeight": 10,
                "itemGap": 18,
                "textStyle": {"fontFamily": FONT_SANS, "fontSize": 18, "color": p["axis"]},
            },
            "tooltip": {
                "trigger": "item",
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
            },
            "xAxis": x_axes,
            "yAxis": y_axes,
            "series": series,
        }
    )
    return EChartsFigure(
        option,
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )
