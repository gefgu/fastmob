from __future__ import annotations

from typing import Any

import numpy as np

from .brand import FONT_MONO, FONT_SANS
from .common import COLOR_ROLES, base_option, norm_width, resolve_palette
from .figure import EChartsFigure

_DASH_PATTERNS: tuple[list[int] | None, ...] = (None, [6, 4], [4, 4], [2, 4], [1, 5])


def _positive_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if (array <= 0).any():
        raise ValueError(f"{name} must contain only positive values")
    return array


def _xy_arrays(x_values: Any, y_values: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = _positive_array(x_values, f"{name} x_points")
    y = _positive_array(y_values, f"{name} y_points")
    if x.shape != y.shape:
        raise ValueError(f"{name} x_points and y_points must have the same shape")
    order = np.argsort(x)
    return x[order], y[order]


def _finite_parameters(values: Any, size: int, name: str) -> tuple[float, ...]:
    parameters = np.asarray(values, dtype=float)
    if parameters.ndim != 1 or parameters.size != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.isfinite(parameters).all():
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(value) for value in parameters)


def _curve_range(all_x: list[np.ndarray], n_points: int, *, logarithmic: bool) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    x_min = min(float(x.min()) for x in all_x)
    x_max = max(float(x.max()) for x in all_x)
    if x_min == x_max:
        return np.asarray([x_min], dtype=float)
    if logarithmic:
        return np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    return np.linspace(x_min, x_max, n_points)


def _points(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    return [[float(x_value), float(y_value)] for x_value, y_value in zip(x, y)]


def _role(index: int) -> str:
    return COLOR_ROLES[min(index, len(COLOR_ROLES) - 1)]


def _dash(index: int) -> list[int] | None:
    return _DASH_PATTERNS[min(index, len(_DASH_PATTERNS) - 1)]


def _scatter_series(name: str, x: np.ndarray, y: np.ndarray, color: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "scatter",
        "data": _points(x, y),
        "symbol": "circle",
        "symbolSize": 9,
        "itemStyle": {
            "color": "transparent",
            "borderColor": color,
            "borderWidth": 2,
        },
        "emphasis": {"scale": 1.35},
        "z": 3,
    }


def _line_series(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    dash: list[int] | None,
    *,
    parameters: dict[str, float],
    reference: bool = False,
) -> dict[str, Any]:
    line_style: dict[str, Any] = {"color": color, "width": 2.5}
    if dash is not None:
        line_style["type"] = dash
    return {
        "name": name,
        "type": "line",
        "data": _points(x, y),
        "showSymbol": False,
        "lineStyle": line_style,
        "itemStyle": {"color": color},
        "fitParameters": parameters,
        "reference": reference,
        "z": 1 if reference else 2,
    }


def _axis(name: str, p: dict, *, logarithmic: bool) -> dict[str, Any]:
    return {
        "type": "log" if logarithmic else "value",
        "logBase": 10,
        "name": name.upper(),
        "nameLocation": "middle",
        "nameGap": 46 if logarithmic else 42,
        "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
        "axisLabel": {"fontFamily": FONT_MONO, "fontSize": 13, "color": p["axis"]},
        "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
        "axisTick": {"lineStyle": {"color": p["axis"]}},
        "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
        "minorSplitLine": {"show": logarithmic, "lineStyle": {"color": p["grid"], "opacity": 0.35}},
    }


def _base_law_option(
    *,
    title: str,
    p: dict,
    chart_type: str,
    x_label: str,
    y_label: str,
    x_log: bool,
) -> dict[str, Any]:
    option = base_option(title, p, chart_type)
    option.update(
        {
            "grid": {"left": 86, "right": 34, "top": 124, "bottom": 76, "containLabel": False},
            "legend": {
                "type": "scroll",
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
            "xAxis": _axis(x_label, p, logarithmic=x_log),
            "yAxis": _axis(y_label, p, logarithmic=True),
            "series": [],
        }
    )
    option["yAxis"]["nameRotate"] = 90
    option["yAxis"]["nameGap"] = 64
    return option


def _figure(
    option: dict[str, Any],
    p: dict,
    width: int | str,
    height: str,
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    return EChartsFigure(
        option,
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def _geometric_scale(y: np.ndarray, shape: np.ndarray) -> float:
    valid = np.isfinite(shape) & (shape > 0)
    if not valid.any():
        raise ValueError("reference curve cannot be scaled over the supplied data")
    return float(np.exp(np.mean(np.log(y[valid]) - np.log(shape[valid]))))


def plot_truncated_powerlaw_fits(
    *datasets: tuple[Any, Any, Any, str],
    title: str = "Truncated power-law fits",
    x_label: str = "travel distance · km",
    y_label: str = "P(Δr)",
    palette: str | dict = "warm",
    show_reference: bool = True,
    reference_parameters: tuple[float, float, float] = (1.5, 1.75, 400.0),
    reference_label: str = "González reference",
    n_points: int = 200,
    width: int | str = 600,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Plot empirical points and precomputed truncated power-law fits."""
    if not datasets:
        raise ValueError("at least one dataset is required")

    normalized = []
    for index, dataset in enumerate(datasets):
        if len(dataset) != 4:
            raise ValueError("each dataset must be (parameters, x_points, y_points, label)")
        parameters, x_values, y_values, label = dataset
        c, r0, beta, kappa = _finite_parameters(parameters, 4, f"dataset {index} parameters")
        if c <= 0 or r0 < 0 or beta <= 0 or kappa <= 0:
            raise ValueError("truncated power-law parameters require c, beta, kappa > 0 and r0 >= 0")
        x, y = _xy_arrays(x_values, y_values, f"dataset {index}")
        normalized.append((c, r0, beta, kappa, x, y, str(label)))

    curve_x = _curve_range([item[4] for item in normalized], n_points, logarithmic=True)
    p = resolve_palette(palette)
    option = _base_law_option(
        title=title, p=p, chart_type="mobility_law", x_label=x_label, y_label=y_label, x_log=True
    )
    for index, (c, r0, beta, kappa, x, y, label) in enumerate(normalized):
        role = _role(index)
        color = p[role]
        curve_y = c * np.power(curve_x + r0, -beta) * np.exp(-curve_x / kappa)
        option["series"].append(_scatter_series(label, x, y, color))
        option["series"].append(
            _line_series(
                f"{label} fit",
                curve_x,
                curve_y,
                color,
                _dash(index),
                parameters={"c": c, "r0": r0, "beta": beta, "kappa": kappa},
            )
        )

    if show_reference:
        r0, beta, kappa = _finite_parameters(reference_parameters, 3, "reference_parameters")
        if r0 < 0 or beta <= 0 or kappa <= 0:
            raise ValueError("reference parameters require beta, kappa > 0 and r0 >= 0")
        all_x = np.concatenate([item[4] for item in normalized])
        all_y = np.concatenate([item[5] for item in normalized])
        shape_at_data = np.power(all_x + r0, -beta) * np.exp(-all_x / kappa)
        c = _geometric_scale(all_y, shape_at_data)
        reference_y = c * np.power(curve_x + r0, -beta) * np.exp(-curve_x / kappa)
        option["series"].append(
            _line_series(
                reference_label,
                curve_x,
                reference_y,
                p["baseline"],
                [6, 4],
                parameters={"c": c, "r0": r0, "beta": beta, "kappa": kappa},
                reference=True,
            )
        )
    return _figure(option, p, width, height, bundle_libs, display)


def plot_lognormal_fits(
    *datasets: tuple[Any, Any, float, float, str],
    title: str = "Daily visited locations",
    x_label: str = "number of visited locations (N)",
    y_label: str = "distribution f(N)",
    palette: str | dict = "warm",
    show_reference: bool = True,
    reference_parameters: tuple[float, float] = (1.0, 0.5),
    reference_label: str = "Log-normal reference",
    n_points: int = 200,
    width: int | str = 600,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Plot empirical points and precomputed log-normal fits."""
    if not datasets:
        raise ValueError("at least one dataset is required")

    normalized = []
    for index, dataset in enumerate(datasets):
        if len(dataset) != 5:
            raise ValueError("each dataset must be (x_points, y_points, mu, sigma, label)")
        x_values, y_values, mu, sigma, label = dataset
        mu, sigma = _finite_parameters((mu, sigma), 2, f"dataset {index} parameters")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        x, y = _xy_arrays(x_values, y_values, f"dataset {index}")
        normalized.append((x, y, mu, sigma, str(label)))

    curve_x = _curve_range([item[0] for item in normalized], n_points, logarithmic=False)
    p = resolve_palette(palette)
    option = _base_law_option(
        title=title, p=p, chart_type="mobility_law", x_label=x_label, y_label=y_label, x_log=False
    )
    for index, (x, y, mu, sigma, label) in enumerate(normalized):
        role = _role(index)
        color = p[role]
        curve_y = np.exp(-((np.log(curve_x) - mu) ** 2) / (2 * sigma**2)) / (
            curve_x * sigma * np.sqrt(2 * np.pi)
        )
        option["series"].append(_scatter_series(label, x, y, color))
        option["series"].append(
            _line_series(
                f"{label} fit",
                curve_x,
                curve_y,
                color,
                _dash(index),
                parameters={"mu": mu, "sigma": sigma},
            )
        )

    if show_reference:
        mu, sigma = _finite_parameters(reference_parameters, 2, "reference_parameters")
        if sigma <= 0:
            raise ValueError("reference sigma must be positive")
        reference_y = np.exp(-((np.log(curve_x) - mu) ** 2) / (2 * sigma**2)) / (
            curve_x * sigma * np.sqrt(2 * np.pi)
        )
        option["series"].append(
            _line_series(
                reference_label,
                curve_x,
                reference_y,
                p["baseline"],
                [6, 4],
                parameters={"mu": mu, "sigma": sigma},
                reference=True,
            )
        )
    return _figure(option, p, width, height, bundle_libs, display)


def plot_distance_frequency_law(
    *datasets: tuple[Any, Any, float, float, str],
    title: str = "Distance-frequency visitation law",
    x_label: str = "r · f · km",
    y_label: str = "ρᵢ(r,f) · visitors km⁻²",
    palette: str | dict = "warm",
    show_reference: bool = True,
    reference_alpha: float = -2.0,
    reference_label: str = "Schläpfer reference",
    n_points: int = 200,
    width: int | str = 600,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Plot empirical distance-frequency points and fitted visitation laws."""
    if not datasets:
        raise ValueError("at least one dataset is required")

    normalized = []
    for index, dataset in enumerate(datasets):
        if len(dataset) != 5:
            raise ValueError("each dataset must be (rf_points, rho_points, eta, mu, label)")
        rf_values, rho_values, eta, mu, label = dataset
        eta, mu = _finite_parameters((eta, mu), 2, f"dataset {index} parameters")
        if eta <= 0 or mu <= 0:
            raise ValueError("eta and mu must be positive")
        x, y = _xy_arrays(rf_values, rho_values, f"dataset {index}")
        normalized.append((x, y, eta, mu, str(label)))

    curve_x = _curve_range([item[0] for item in normalized], n_points, logarithmic=True)
    p = resolve_palette(palette)
    option = _base_law_option(
        title=title, p=p, chart_type="mobility_law", x_label=x_label, y_label=y_label, x_log=True
    )
    for index, (x, y, eta, mu, label) in enumerate(normalized):
        role = _role(index)
        color = p[role]
        curve_y = mu * np.power(curve_x, -eta)
        option["series"].append(_scatter_series(label, x, y, color))
        option["series"].append(
            _line_series(
                f"{label} fit",
                curve_x,
                curve_y,
                color,
                _dash(index),
                parameters={"eta": eta, "mu": mu},
            )
        )

    reference_alpha = float(reference_alpha)
    if show_reference:
        if not np.isfinite(reference_alpha):
            raise ValueError("reference_alpha must be finite")
        all_x = np.concatenate([item[0] for item in normalized])
        all_y = np.concatenate([item[1] for item in normalized])
        shape_at_data = np.power(all_x, reference_alpha)
        scale = _geometric_scale(all_y, shape_at_data)
        option["series"].append(
            _line_series(
                reference_label,
                curve_x,
                scale * np.power(curve_x, reference_alpha),
                p["baseline"],
                [6, 4],
                parameters={"alpha": reference_alpha, "scale": scale},
                reference=True,
            )
        )
    return _figure(option, p, width, height, bundle_libs, display)
