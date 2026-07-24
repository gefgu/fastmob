from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from ._core import (
    activity_time_bin_labels,
    finalize_activity_bar_option_json,
    finalize_activity_comparison_option_json,
    finalize_activity_heatmap_option_json,
)
from .brand import FONT_MONO, FONT_SANS
from .common import activity_color, base_option, heatmap_option, norm_width, resolve_palette
from .data import (
    columns,
    daily_from_raw,
    dataframe_like,
    distribution_from_precomputed,
    distribution_from_raw,
    matrix_from_precomputed,
    transition_from_raw,
)
from .figure import EChartsFigure


def _time_bin_labels(n_bins: int) -> list[str]:
    return activity_time_bin_labels(n_bins)


def _validate_labels(labels: Sequence[str]) -> tuple[str, str]:
    if isinstance(labels, (str, bytes)) or len(labels) != 2:
        raise ValueError("labels must contain exactly two dataset names")
    return str(labels[0]), str(labels[1])


def _validate_categories(categories: Sequence[str], name: str) -> list[str]:
    normalized = [str(category) for category in categories]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate activity labels")
    return normalized


def _normalize_transition_data(data: Any, name: str) -> tuple[list[str], np.ndarray]:
    data_columns = columns(data)
    is_labeled_matrix = (
        hasattr(data, "index")
        and not callable(data.index)
        and hasattr(data, "columns")
        and hasattr(data, "values")
    )
    index_labels = [str(label) for label in data.index.tolist()] if is_labeled_matrix else []
    column_labels = [str(label) for label in data_columns]
    looks_like_square_matrix = (
        is_labeled_matrix
        and len(index_labels) == len(column_labels)
        and set(index_labels) == set(column_labels)
    )
    if dataframe_like(data) and not looks_like_square_matrix:
        categories, matrix = transition_from_raw(data)
    elif looks_like_square_matrix:
        categories = index_labels
        column_indexes = {category: index for index, category in enumerate(column_labels)}
        matrix = np.asarray(data.values, dtype=float)[:, [column_indexes[category] for category in categories]]
    else:
        categories, matrix = matrix_from_precomputed(data)

    categories = _validate_categories(categories, name)
    values = np.asarray(matrix, dtype=float)
    expected_shape = (len(categories), len(categories))
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError(f"{name} transition matrix must have shape {expected_shape}")
    if np.isinf(values).any():
        raise ValueError(f"{name} transition matrix must contain finite values or NaN")
    return categories, values


def _normalize_daily_data(
    data: Any,
    *,
    categories: Sequence[str] | None,
    n_bins: int | None,
    name: str,
) -> tuple[np.ndarray, list[str], int]:
    if isinstance(data, tuple) and len(data) == 3:
        matrix, categories_from_data, n_bins_from_data = data
        values = np.asarray(matrix, dtype=float)
        normalized_categories = _validate_categories(categories_from_data, name)
        normalized_n_bins = int(n_bins_from_data)
    elif categories is not None and n_bins is not None:
        values = np.asarray(data, dtype=float)
        normalized_categories = _validate_categories(categories, name)
        normalized_n_bins = int(n_bins)
    elif dataframe_like(data):
        matrix, categories_from_data, n_bins_from_data = daily_from_raw(data)
        values = np.asarray(matrix, dtype=float)
        normalized_categories = _validate_categories(categories_from_data, name)
        normalized_n_bins = int(n_bins_from_data)
    else:
        raise ValueError("daily activity data must be raw visits, a matrix with categories/n_bins, or a tuple")

    expected_shape = (len(normalized_categories), normalized_n_bins)
    if values.ndim != 2 or values.shape != expected_shape:
        raise ValueError(f"{name} daily activity matrix must have shape {expected_shape}")
    if np.isinf(values).any():
        raise ValueError(f"{name} daily activity matrix must contain finite values or NaN")
    _time_bin_labels(normalized_n_bins)
    return values, normalized_categories, normalized_n_bins


def _category_union(first: Sequence[str], second: Sequence[str]) -> list[str]:
    categories = list(first)
    seen = set(categories)
    for category in second:
        if category not in seen:
            categories.append(category)
            seen.add(category)
    return categories


def _align_square_matrix(
    matrix: np.ndarray,
    source_categories: Sequence[str],
    target_categories: Sequence[str],
) -> np.ndarray:
    aligned = np.zeros((len(target_categories), len(target_categories)), dtype=float)
    target_indexes = {category: index for index, category in enumerate(target_categories)}
    source_indexes = [target_indexes[category] for category in source_categories]
    aligned[np.ix_(source_indexes, source_indexes)] = matrix
    return aligned


def _align_daily_matrix(
    matrix: np.ndarray,
    source_categories: Sequence[str],
    target_categories: Sequence[str],
) -> np.ndarray:
    aligned = np.zeros((len(target_categories), matrix.shape[1]), dtype=float)
    target_indexes = {category: index for index, category in enumerate(target_categories)}
    for source_index, category in enumerate(source_categories):
        aligned[target_indexes[category]] = matrix[source_index]
    return aligned


def _difference_limit(values: np.ndarray) -> float:
    finite_values = np.abs(values[np.isfinite(values)])
    return max(float(finite_values.max()) if finite_values.size else 0.0, 1.0)


def _configure_difference_heatmap(
    option: dict[str, Any],
    *,
    p: dict,
    labels: tuple[str, str],
    limit: float,
) -> None:
    first_label, second_label = labels
    option["_meta"]["differenceLabels"] = [first_label, second_label]
    option["visualMap"].update(
        {
            "min": -limit,
            "max": limit,
            "text": [f"{second_label} higher", f"{first_label} higher"],
            "inRange": {"color": [p["observed"], p["bg"], p["synthetic"]]},
        }
    )
    option["series"][0]["name"] = f"{second_label} - {first_label}"


def _visit_purpose_distribution(data: Any) -> tuple[list[str], list[float], list[int], bool]:
    data_columns = columns(data)
    is_precomputed = not (
        dataframe_like(data)
        and not ("activity" in data_columns and ("percentage" in data_columns or "count" in data_columns))
    )
    if is_precomputed:
        categories, percentages, counts = distribution_from_precomputed(data)
        has_counts = (
            (isinstance(data, tuple) and len(data) > 2)
            or "count" in data_columns
        )
    else:
        categories, percentages, counts = distribution_from_raw(data)
        has_counts = True
    return categories, percentages, counts, has_counts


def plot_visit_purpose_distribution(
    data: Any,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Visit Purpose Distribution",
    palette: str | dict = "warm",
    width: int | str = 600,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a bar chart of visit-purpose percentages."""
    p = resolve_palette(palette)
    categories, percentages, counts, _ = _visit_purpose_distribution(data)
    if labels is not None:
        if len(labels) != len(categories):
            raise ValueError("labels length must match the number of categories")
        categories = [str(label) for label in labels]

    option = base_option(title, p, "bar")
    option.update(
        {
            "grid": {"left": 70, "right": 28, "top": 76, "bottom": 76, "containLabel": False},
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
                "formatter": "{b}<br/>{a}: {c}%",
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"fontFamily": FONT_SANS, "fontSize": 13, "color": p["axis"], "interval": 0},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "name": "% OF VISITS",
                "nameLocation": "middle",
                "nameGap": 48,
                "nameRotate": 90,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {"formatter": "{value}", "fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
            },
            "series": [
                {
                    "name": "Percentage",
                    "type": "bar",
                    "barMaxWidth": 42,
                    "data": [],
                    "label": {
                        "show": True,
                        "position": "top",
                        "formatter": "{c}%",
                        "fontFamily": FONT_MONO,
                        "fontSize": 12,
                        "color": p["axis"],
                    },
                }
            ],
        }
    )
    percentages_array = np.ascontiguousarray(percentages, dtype=float)
    option_json = finalize_activity_bar_option_json(
        json.dumps(option),
        percentages_array,
        counts,
        [activity_color(p, i) for i in range(len(percentages))],
    )
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_visit_purpose_comparison(
    datasets: Mapping[str, Any],
    *,
    title: str = "Visit Purpose Comparison",
    palette: str | dict = "warm",
    width: int | str = 800,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a grouped bar chart comparing visit-purpose distributions."""
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("datasets must be a non-empty mapping of labels to visit-purpose data")

    p = resolve_palette(palette)
    dataset_names: list[str] = []
    categories: list[str] = []
    category_indexes: dict[str, int] = {}
    distributions: list[tuple[list[str], list[float], list[int], bool]] = []

    for dataset_name, data in datasets.items():
        name = str(dataset_name)
        dataset_categories, percentages, counts, has_counts = _visit_purpose_distribution(data)
        if len(set(dataset_categories)) != len(dataset_categories):
            raise ValueError(f"dataset {name!r} contains duplicate purpose labels")
        dataset_names.append(name)
        distributions.append((dataset_categories, percentages, counts, has_counts))
        for category in dataset_categories:
            if category not in category_indexes:
                category_indexes[category] = len(categories)
                categories.append(category)

    percentages_matrix = np.zeros((len(distributions), len(categories)), dtype=float)
    aligned_counts: list[list[int | None]] = [
        [None] * len(categories) for _ in distributions
    ]
    for dataset_index, (dataset_categories, percentages, counts, has_counts) in enumerate(distributions):
        for source_index, category in enumerate(dataset_categories):
            category_index = category_indexes[category]
            percentages_matrix[dataset_index, category_index] = percentages[source_index]
            if has_counts:
                aligned_counts[dataset_index][category_index] = counts[source_index]

    opacities = (
        [1.0]
        if len(distributions) == 1
        else np.linspace(1.0, 0.4, len(distributions)).tolist()
    )
    option = base_option(title, p, "visit_purpose_comparison")
    option.update(
        {
            "legend": {
                "data": dataset_names,
                "right": 0,
                "top": 12,
                "itemWidth": 18,
                "itemHeight": 10,
                "itemGap": 18,
                "textStyle": {"fontFamily": FONT_SANS, "fontSize": 13, "color": p["axis"]},
            },
            "grid": {"left": 70, "right": 28, "top": 76, "bottom": 76, "containLabel": False},
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {"fontFamily": FONT_SANS, "fontSize": 13, "color": p["axis"], "interval": 0},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "name": "% OF VISITS",
                "nameLocation": "middle",
                "nameGap": 48,
                "nameRotate": 90,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {"formatter": "{value}", "fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
            },
            "series": [
                {
                    "name": name,
                    "type": "bar",
                    "barMaxWidth": 42,
                    "data": [],
                    "label": {"show": False},
                }
                for name in dataset_names
            ],
        }
    )
    option_json = finalize_activity_comparison_option_json(
        json.dumps(option),
        np.ascontiguousarray(percentages_matrix),
        aligned_counts,
        [activity_color(p, index) for index in range(len(categories))],
        opacities,
    )
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_activity_transition_matrix(
    matrix_or_visits: Any,
    *,
    title: str = "Activity Transition Matrix",
    palette: str | dict = "warm",
    width: int | str = 600,
    height: str = "520px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a heatmap of activity transitions."""
    p = resolve_palette(palette)
    categories, values = _normalize_transition_data(matrix_or_visits, "data")
    option = heatmap_option(
        title=title,
        p=p,
        chart_type="transition",
        x_labels=categories,
        y_labels=categories,
        data=[],
        vmax=100.0,
        x_name="TO ACTIVITY",
        y_name="FROM ACTIVITY",
        tooltip_value_suffix="%",
        grid={"left": 92, "right": 96, "top": 80, "bottom": 86, "containLabel": False},
    )
    values = np.ascontiguousarray(values, dtype=float)
    option_json = finalize_activity_heatmap_option_json(json.dumps(option), values, True)
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_activity_transition_difference(
    first: Any,
    second: Any,
    *,
    labels: Sequence[str] = ("first", "second"),
    title: str = "Activity Transition Difference",
    palette: str | dict = "warm",
    width: int | str = 600,
    height: str = "520px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a signed heatmap of second-minus-first transition percentages."""
    p = resolve_palette(palette)
    normalized_labels = _validate_labels(labels)
    first_categories, first_values = _normalize_transition_data(first, normalized_labels[0])
    second_categories, second_values = _normalize_transition_data(second, normalized_labels[1])
    categories = _category_union(first_categories, second_categories)
    difference = _align_square_matrix(
        second_values, second_categories, categories
    ) - _align_square_matrix(first_values, first_categories, categories)
    limit = _difference_limit(difference)

    option = heatmap_option(
        title=title,
        p=p,
        chart_type="transition_difference",
        x_labels=categories,
        y_labels=categories,
        data=[],
        vmax=limit,
        x_name="TO ACTIVITY",
        y_name="FROM ACTIVITY",
        tooltip_value_suffix=" pp",
        grid={"left": 92, "right": 118, "top": 80, "bottom": 86, "containLabel": False},
    )
    option["series"][0]["label"]["show"] = True
    option["series"][0]["label"]["fontSize"] = 16
    _configure_difference_heatmap(option, p=p, labels=normalized_labels, limit=limit)
    option_json = finalize_activity_heatmap_option_json(
        json.dumps(option), np.ascontiguousarray(difference, dtype=float)
    )
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_daily_activity_distribution(
    data: Any,
    *,
    categories: Sequence[str] | None = None,
    n_bins: int | None = None,
    title: str = "Daily Activity Distribution",
    palette: str | dict = "warm",
    width: int | str = 900,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a DARD heatmap of activity percentages by time of day."""
    p = resolve_palette(palette)
    matrix, normalized_categories, normalized_n_bins = _normalize_daily_data(
        data, categories=categories, n_bins=n_bins, name="data"
    )

    x_labels = _time_bin_labels(normalized_n_bins)
    option = heatmap_option(
        title=title,
        p=p,
        chart_type="daily_activity",
        x_labels=x_labels,
        y_labels=normalized_categories,
        data=[],
        vmax=100.0,
        x_name="TIME OF DAY",
        y_name="",
        tooltip_value_suffix="%",
        grid={"left": 96, "right": 96, "top": 80, "bottom": 76, "containLabel": False},
    )
    option["xAxis"]["axisLabel"]["interval"] = max((normalized_n_bins // 24) - 1, 0)
    option["xAxis"]["axisLabel"]["rotate"] = 45
    matrix = np.ascontiguousarray(matrix, dtype=float)
    option_json = finalize_activity_heatmap_option_json(json.dumps(option), matrix)
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )


def plot_daily_activity_difference(
    first: Any,
    second: Any,
    *,
    categories: Sequence[str] | None = None,
    n_bins: int | None = None,
    labels: Sequence[str] = ("first", "second"),
    title: str = "Daily Activity Difference",
    palette: str | dict = "warm",
    width: int | str = 900,
    height: str = "420px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a signed heatmap of second-minus-first daily activity percentages."""
    p = resolve_palette(palette)
    normalized_labels = _validate_labels(labels)
    first_values, first_categories, first_n_bins = _normalize_daily_data(
        first, categories=categories, n_bins=n_bins, name=normalized_labels[0]
    )
    second_values, second_categories, second_n_bins = _normalize_daily_data(
        second, categories=categories, n_bins=n_bins, name=normalized_labels[1]
    )
    if first_n_bins != second_n_bins:
        raise ValueError(
            f"daily activity datasets must use the same n_bins; got {first_n_bins} and {second_n_bins}"
        )

    aligned_categories = _category_union(first_categories, second_categories)
    difference = _align_daily_matrix(
        second_values, second_categories, aligned_categories
    ) - _align_daily_matrix(first_values, first_categories, aligned_categories)
    limit = _difference_limit(difference)
    x_labels = _time_bin_labels(first_n_bins)
    option = heatmap_option(
        title=title,
        p=p,
        chart_type="daily_activity_difference",
        x_labels=x_labels,
        y_labels=aligned_categories,
        data=[],
        vmax=limit,
        x_name="TIME OF DAY",
        y_name="",
        tooltip_value_suffix=" pp",
        grid={"left": 96, "right": 118, "top": 80, "bottom": 76, "containLabel": False},
    )
    option["xAxis"]["axisLabel"]["interval"] = max((first_n_bins // 24) - 1, 0)
    option["xAxis"]["axisLabel"]["rotate"] = 45
    _configure_difference_heatmap(option, p=p, labels=normalized_labels, limit=limit)
    option_json = finalize_activity_heatmap_option_json(
        json.dumps(option), np.ascontiguousarray(difference, dtype=float)
    )
    return EChartsFigure(
        json.loads(option_json),
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )
