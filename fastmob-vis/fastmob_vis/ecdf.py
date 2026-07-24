from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ._core import build_ecdf_option_json, compute_ecdf
from .brand import FONT_MONO, FONT_SANS, FONT_SERIF
from .common import norm_width, resolve_palette
from .figure import EChartsFigure

SERIES_ROLES = ["observed", "synthetic", "syn2", "syn3", "baseline"]
DASH_PATTERNS: dict[str, list[int] | None] = {
    "observed": None,
    "synthetic": [6, 4],
    "syn2": [4, 4],
    "syn3": [2, 4],
    "baseline": [1, 5],
}


def _as_float_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _build_series_inputs(
    label_points: list[tuple[str, list]],
    p: dict,
) -> list[tuple[str, list, str, list[int] | None]]:
    result = []
    for i, (name, points) in enumerate(label_points):
        role = SERIES_ROLES[min(i, len(SERIES_ROLES) - 1)]
        color = p[role]
        dash = DASH_PATTERNS[role]
        result.append((name, points, color, dash))
    return result


def _plot_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str],
    title: str,
    palette: str | dict,
    cdf_cutoff: float,
    x_label: str,
    x_unit: str,
    width: int | str,
    bundle_libs: bool,
    display: str,
) -> EChartsFigure:
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings")

    p = resolve_palette(palette)
    left_arr = _as_float_array(left, "left")
    right_arr = _as_float_array(right, "right")
    left_points = compute_ecdf(left_arr, cdf_cutoff)
    right_points = compute_ecdf(right_arr, cdf_cutoff)
    x_name = f"{x_label.upper()} · {x_unit.upper()}" if x_unit else x_label.upper()

    series_inputs = _build_series_inputs(
        [(str(labels[0]), left_points), (str(labels[1]), right_points)], p
    )
    option_json = build_ecdf_option_json(
        series_inputs, title, x_name, x_unit, x_label,
        p["bg"], p["axis"], p["grid"],
        FONT_SANS, FONT_SERIF, FONT_MONO,
    )
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_jump_lengths_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str] = ("A", "B"),
    title: str = "Jump length ECDF",
    palette: str | dict = "warm",
    cdf_cutoff: float = 0.98,
    x_label: str = "jump length",
    x_unit: str = "km",
    width: int | str = 600,
    *,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a Jupyter-renderable ECDF comparison for two jump-length arrays."""
    return _plot_ecdf(left, right, labels, title, palette, cdf_cutoff, x_label, x_unit, width, bundle_libs, display)


def plot_visits_frequency_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str] = ("A", "B"),
    title: str = "Visits frequency ECDF",
    palette: str | dict = "warm",
    cdf_cutoff: float = 0.98,
    x_label: str = "number of visits",
    x_unit: str = "",
    width: int | str = 600,
    *,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a Jupyter-renderable ECDF comparison for two visit-frequency arrays."""
    return _plot_ecdf(left, right, labels, title, palette, cdf_cutoff, x_label, x_unit, width, bundle_libs, display)


def plot_radius_of_gyration_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str] = ("A", "B"),
    title: str = "Radius of gyration ECDF",
    palette: str | dict = "warm",
    cdf_cutoff: float = 0.98,
    x_label: str = "radius of gyration",
    x_unit: str = "km",
    width: int | str = 600,
    *,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a Jupyter-renderable ECDF comparison for two radius-of-gyration arrays."""
    return _plot_ecdf(left, right, labels, title, palette, cdf_cutoff, x_label, x_unit, width, bundle_libs, display)


def plot_trip_duration_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str] = ("A", "B"),
    title: str = "Trip duration ECDF",
    palette: str | dict = "warm",
    cdf_cutoff: float = 0.98,
    x_label: str = "trip duration",
    x_unit: str = "min",
    width: int | str = 600,
    *,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a Jupyter-renderable ECDF comparison for two trip-duration arrays."""
    return _plot_ecdf(left, right, labels, title, palette, cdf_cutoff, x_label, x_unit, width, bundle_libs, display)


def plot_dwell_time_ecdf(
    left: Any,
    right: Any,
    labels: Sequence[str] = ("A", "B"),
    title: str = "Dwell time ECDF",
    palette: str | dict = "warm",
    cdf_cutoff: float = 0.98,
    x_label: str = "dwell time",
    x_unit: str = "min",
    width: int | str = 600,
    *,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a Jupyter-renderable ECDF comparison for two dwell-time arrays."""
    return _plot_ecdf(left, right, labels, title, palette, cdf_cutoff, x_label, x_unit, width, bundle_libs, display)
