from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from fastmob.utils._common import _values_to_list as series_to_list


def columns(data: Any) -> list[str]:
    if isinstance(data, dict):
        return list(data)
    if hasattr(data, "columns"):
        return list(data.columns)
    return []


def column(data: Any, name: str) -> list:
    if isinstance(data, dict):
        return series_to_list(data[name])
    return series_to_list(data[name])


def pick_column(data: Any, candidates: Sequence[str]) -> str | None:
    available = set(columns(data))
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def data_len(data: Any) -> int:
    if isinstance(data, dict):
        if not data:
            return 0
        return len(next(iter(data.values())))
    return len(data)


def dataframe_like(data: Any) -> bool:
    return isinstance(data, dict) or hasattr(data, "columns")


def _metric_input(data: Any) -> Any:
    if isinstance(data, dict):
        import pandas as pd

        return pd.DataFrame(data)
    return data


def distribution_from_raw(data: Any, unknown_label: str = "UNKNOWN") -> tuple[list[str], list[float], list[int]]:
    from fastmob import visit_purpose_distribution

    return distribution_from_precomputed(
        visit_purpose_distribution(_metric_input(data), unknown_label=unknown_label)
    )


def distribution_from_precomputed(data: Any) -> tuple[list[str], list[float], list[int]]:
    if isinstance(data, tuple) and len(data) >= 2:
        labels = [str(label) for label in data[0]]
        percentages = [float(value) for value in data[1]]
        counts = [int(value) for value in data[2]] if len(data) > 2 else [0] * len(labels)
        return labels, percentages, counts

    data_columns = columns(data)
    if "activity" in data_columns and ("percentage" in data_columns or "count" in data_columns):
        labels = [str(label) for label in column(data, "activity")]
        counts = [int(value) for value in column(data, "count")] if "count" in data_columns else [0] * len(labels)
        if "percentage" in data_columns:
            percentages = [float(value) for value in column(data, "percentage")]
        else:
            total = sum(counts)
            percentages = [(count / total) * 100.0 if total else 0.0 for count in counts]
        return labels, percentages, counts

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        pairs = list(data)
        if pairs and isinstance(pairs[0], Sequence):
            labels = [str(pair[0]) for pair in pairs]
            percentages = [float(pair[1]) for pair in pairs]
            return labels, percentages, [0] * len(labels)

    raise ValueError("distribution data must be raw visits or contain activity/count/percentage values")


def transition_from_raw(data: Any, unknown_label: str = "UNKNOWN") -> tuple[list[str], list[list[float]]]:
    from fastmob import activity_transition_matrix

    return matrix_from_precomputed(
        activity_transition_matrix(_metric_input(data), unknown_label=unknown_label)
    )


def matrix_from_precomputed(data: Any) -> tuple[list[str], list[list[float]]]:
    if hasattr(data, "index") and hasattr(data, "columns") and hasattr(data, "values"):
        return [str(label) for label in data.index.tolist()], np.asarray(data.values, dtype=float).tolist()
    if isinstance(data, tuple) and len(data) == 2:
        labels, matrix = data
        return [str(label) for label in labels], np.asarray(matrix, dtype=float).tolist()
    data_columns = columns(data)
    if "activity" in data_columns:
        labels = [str(label) for label in column(data, "activity")]
        matrix = [[float(value) for value in column(data, label)] for label in labels]
        return labels, np.asarray(matrix, dtype=float).T.tolist()
    return [str(i) for i in range(len(data))], np.asarray(data, dtype=float).tolist()


def daily_from_raw(data: Any, unknown_label: str = "UNKNOWN") -> tuple[list[list[float]], list[str], int]:
    from fastmob import daily_activity_distribution

    matrix, categories, n_bins = daily_activity_distribution(
        _metric_input(data), unknown_label=unknown_label
    )
    return np.asarray(matrix, dtype=float).tolist(), [str(category) for category in categories], int(n_bins)
