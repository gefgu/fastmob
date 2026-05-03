"""Comparison metrics for mobility distributions and profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

import narwhals as nw
import numpy as np

from skmob2._core import wasserstein_numpy as _wasserstein_numpy

from ._common import (
    ACTIVITY_CANDIDATES,
    DAY_CANDIDATES,
    DURATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _pick_existing_column,
)
from .visits.motifs import discover_daily_motifs_from_agents

_DAY_PERIOD_COLUMN = "day_period"
_WEEKDAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_PERIOD_ORDER = [
    "Morning (06:00-11:59)",
    "Afternoon (12:00-17:59)",
    "Evening (18:00-23:59)",
    "Night (00:00-05:59)",
]


def _simsimd_jensenshannon(p: np.ndarray, q: np.ndarray) -> float:
    try:
        import simsimd
    except ImportError as exc:
        raise ImportError("simsimd is required for Jensen-Shannon metrics: pip install simsimd") from exc
    return float(simsimd.jensenshannon(p, q))


def _reference_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log(p / m), 0.0)
        right = np.where(q > 0, q * np.log(q / m), 0.0)
    return float(0.5 * (np.sum(left) + np.sum(right)))


def _normalize_distribution(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = np.nan_to_num(arr, nan=0.0)
    total = float(arr.sum())
    if total > 0.0:
        arr = arr / total
    return arr


def jensen_shannon_divergence(distribution1: Any, distribution2: Any) -> float:
    """Return Jensen-Shannon divergence between two distributions.

    The result matches the historical comparison behavior based on
    ``scipy.spatial.distance.jensenshannon(...) ** 2``.
    """
    p = _normalize_distribution(distribution1)
    q = _normalize_distribution(distribution2)
    if p.shape != q.shape:
        raise ValueError(f"distribution shapes must match, got {p.shape} and {q.shape}")
    if p.size == 0 or (p.sum() == 0.0 and q.sum() == 0.0):
        return 0.0
    raw = _simsimd_jensenshannon(p, q)
    expected = _reference_js_divergence(p, q)
    if np.isclose(raw * raw, expected, rtol=1e-10, atol=1e-12):
        return float(raw * raw)
    if np.isclose(raw, expected, rtol=1e-10, atol=1e-12):
        return float(raw)
    return expected


def _matrix_values_and_labels(matrix: Any, categories: list[Any] | None) -> tuple[np.ndarray, list[Any] | None]:
    if categories is not None:
        return np.asarray(matrix, dtype=np.float64), list(categories)
    if hasattr(matrix, "values") and hasattr(matrix, "index"):
        return np.asarray(matrix.values, dtype=np.float64), list(matrix.index)
    return np.asarray(matrix, dtype=np.float64), None


def _align_category_matrix(
    matrix1: Any,
    matrix2: Any,
    categories1: list[Any] | None = None,
    categories2: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    m1, c1 = _matrix_values_and_labels(matrix1, categories1)
    m2, c2 = _matrix_values_and_labels(matrix2, categories2)
    if m1.shape == m2.shape and (c1 is None or c2 is None or c1 == c2):
        return m1, m2
    if c1 is None or c2 is None:
        raise ValueError(
            f"Matrix shapes must match. Got {m1.shape} and {m2.shape}. "
            "Provide categories1 and categories2 to align different activity sets."
        )
    labels = sorted(set(c1) | set(c2), key=lambda value: str(value))
    idx = {label: i for i, label in enumerate(labels)}
    out1 = np.zeros((len(labels), len(labels)), dtype=np.float64)
    out2 = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for r, from_label in enumerate(c1):
        for c, to_label in enumerate(c1):
            out1[idx[from_label], idx[to_label]] = m1[r, c]
    for r, from_label in enumerate(c2):
        for c, to_label in enumerate(c2):
            out2[idx[from_label], idx[to_label]] = m2[r, c]
    return out1, out2


def matrix_jensen_shannon_divergence(
    matrix1: Any,
    matrix2: Any,
    categories1: list[Any] | None = None,
    categories2: list[Any] | None = None,
) -> float:
    """Return Jensen-Shannon divergence between two category matrices."""
    m1, m2 = _align_category_matrix(matrix1, matrix2, categories1, categories2)
    return jensen_shannon_divergence(m1.ravel(), m2.ravel())


def time_bin_matrix_jensen_shannon_divergence(
    matrix1: Any,
    matrix2: Any,
    categories1: list[Any] | None = None,
    categories2: list[Any] | None = None,
) -> float:
    """Return mean per-column Jensen-Shannon divergence for time-bin matrices."""
    m1 = np.asarray(matrix1, dtype=np.float64)
    m2 = np.asarray(matrix2, dtype=np.float64)
    if m1.shape[1] != m2.shape[1]:
        raise ValueError(f"Number of time bins must match. Got {m1.shape[1]} and {m2.shape[1]}")
    if m1.shape[0] != m2.shape[0]:
        if categories1 is None or categories2 is None:
            raise ValueError(
                f"Matrix shapes must match. Got {m1.shape} and {m2.shape}. "
                "Provide categories1 and categories2 to align matrices with different activity sets."
            )
        labels = sorted(set(categories1) | set(categories2), key=lambda value: str(value))
        idx = {label: i for i, label in enumerate(labels)}
        a1 = np.zeros((len(labels), m1.shape[1]), dtype=np.float64)
        a2 = np.zeros((len(labels), m2.shape[1]), dtype=np.float64)
        for row, label in enumerate(categories1):
            a1[idx[label], :] = m1[row, :]
        for row, label in enumerate(categories2):
            a2[idx[label], :] = m2[row, :]
        m1, m2 = a1, a2
    elif m1.shape != m2.shape:
        raise ValueError(f"Matrix shapes must match. Got {m1.shape} and {m2.shape}.")

    values = []
    for col in range(m1.shape[1]):
        left = np.nan_to_num(m1[:, col], nan=0.0)
        right = np.nan_to_num(m2[:, col], nan=0.0)
        if left.sum() == 0.0 and right.sum() == 0.0:
            continue
        values.append(jensen_shannon_divergence(left, right))
    return 0.0 if not values else float(np.mean(values))


def histogram_jensen_shannon_divergence(values1: Any, values2: Any, bin_size: float = 1.0) -> float:
    """Bin two value arrays and return Jensen-Shannon divergence."""
    v1 = _finite_array(values1)
    v2 = _finite_array(values2)
    if v1.size == 0 or v2.size == 0:
        return float("nan")
    max_value = float(max(v1.max(), v2.max()))
    if max_value <= 0.0:
        bins = np.array([0.0, float(bin_size)])
    else:
        bins = np.arange(0.0, max_value + bin_size, bin_size, dtype=np.float64)
        if bins.size < 2:
            bins = np.array([0.0, float(bin_size)])
    hist1, _ = np.histogram(v1, bins=bins)
    hist2, _ = np.histogram(v2, bins=bins)
    return jensen_shannon_divergence(hist1, hist2)


def _finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def wasserstein_distance(values1: Any, values2: Any) -> float:
    """Return Rust-backed 1D Wasserstein distance between empirical samples."""
    v1 = _finite_array(values1)
    v2 = _finite_array(values2)
    if v1.size == 0 or v2.size == 0:
        return float("nan")
    return float(_wasserstein_numpy(np.ascontiguousarray(v1, dtype=np.float64), np.ascontiguousarray(v2, dtype=np.float64)))


def _series_values(df: nw.DataFrame, column: str) -> list[Any]:
    return df.get_column(column).to_list()


def _is_null(value: Any) -> bool:
    try:
        return bool(value is None or value != value)
    except TypeError:
        return value is None


def _hour_from_value(value: Any) -> int | None:
    if _is_null(value):
        return None
    if hasattr(value, "hour"):
        return int(value.hour)
    text = str(value)
    try:
        if ":" in text:
            return int(text.split(":", 1)[0][-2:])
        return int(np.datetime64(text, "h").astype(object).hour)
    except Exception:
        return None


def _day_period(value: Any) -> str | None:
    hour = _hour_from_value(value)
    if hour is None:
        return None
    if 6 <= hour < 12:
        return _DAY_PERIOD_ORDER[0]
    if 12 <= hour < 18:
        return _DAY_PERIOD_ORDER[1]
    if 18 <= hour < 24:
        return _DAY_PERIOD_ORDER[2]
    return _DAY_PERIOD_ORDER[3]


def _resolve_column(df: nw.DataFrame, explicit: str | None, candidates: list[str], role: str) -> str:
    if explicit is not None and explicit in df.columns:
        return explicit
    column = _pick_existing_column(df.columns, candidates)
    if column is None:
        tried = [explicit] + candidates if explicit is not None else candidates
        raise ValueError(f"Could not detect {role} column. Tried: {tried}. Available columns: {df.columns}.")
    return column


def _group_labels(groups1: dict[Any, Any], groups2: dict[Any, Any], hue: str | None) -> list[Any]:
    labels = set(groups1) | set(groups2)
    if hue == "day_of_week":
        return [day for day in _WEEKDAY_ORDER if day in labels]
    if hue == "day_period":
        return [period for period in _DAY_PERIOD_ORDER if period in labels]
    if hue == "purpose":
        counts = Counter()
        for groups in (groups1, groups2):
            for label, values in groups.items():
                counts[label] += len(values)
        return [label for label, _ in counts.most_common()]
    return ["All Trips"] if "All Trips" in labels else sorted(labels, key=lambda value: str(value))


def _group_numeric_values(
    data: Any,
    value_col: str,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_col: str | None = None,
    day_col: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> dict[Any, list[float]]:
    df = nw.from_native(data, eager_only=True)
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not found. Available columns: {df.columns}.")
    values = _series_values(df, value_col)

    if hue is None:
        labels = ["All Trips"] * len(values)
    elif hue == "day_of_week":
        day_col = _resolve_column(df, day_col, DAY_CANDIDATES, "day")
        labels = [None if _is_null(v) else str(v).lower() for v in _series_values(df, day_col)]
    elif hue == "day_period":
        if skip_day_period_creation and _DAY_PERIOD_COLUMN in df.columns:
            labels = _series_values(df, _DAY_PERIOD_COLUMN)
        else:
            trip_start_col = _resolve_column(df, trip_start_col, TIMESTAMP_CANDIDATES, "timestamp")
            labels = [_day_period(value) for value in _series_values(df, trip_start_col)]
    elif hue == "purpose":
        purpose_col = _resolve_column(df, purpose_col, ACTIVITY_CANDIDATES, "purpose")
        labels = _series_values(df, purpose_col)
    else:
        raise ValueError(f"Unsupported hue value: {hue!r}")

    groups: dict[Any, list[float]] = defaultdict(list)
    for label, value in zip(labels, values):
        if _is_null(label) or _is_null(value):
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            groups[label].append(value_float)
    return dict(groups)


def column_distribution_jensen_shannon_divergence(
    df1: Any,
    df2: Any,
    column: str,
    *,
    bin_size: float = 1.0,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_col1: str | None = None,
    trip_start_col2: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped numeric column distributions with Jensen-Shannon divergence."""
    groups1 = _group_numeric_values(
        df1,
        column,
        hue=hue,
        trip_start_col=trip_start_col1,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _group_numeric_values(
        df2,
        column,
        hue=hue,
        trip_start_col=trip_start_col2,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, histogram_jensen_shannon_divergence(groups1[label], groups2[label], bin_size=bin_size))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2 and len(groups1[label]) > 0 and len(groups2[label]) > 0
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def column_distribution_wasserstein_distance(
    df1: Any,
    df2: Any,
    column: str,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_col1: str | None = None,
    trip_start_col2: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped numeric column distributions with Rust-backed Wasserstein distance."""
    groups1 = _group_numeric_values(
        df1,
        column,
        hue=hue,
        trip_start_col=trip_start_col1,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _group_numeric_values(
        df2,
        column,
        hue=hue,
        trip_start_col=trip_start_col2,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, wasserstein_distance(groups1[label], groups2[label]))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2 and len(groups1[label]) > 0 and len(groups2[label]) > 0
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def _visits_per_user_groups(
    data: Any,
    *,
    user_id_col: str | None = None,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_col: str | None = None,
    day_col: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> dict[Any, list[int]]:
    df = nw.from_native(data, eager_only=True)
    user_id_col = _resolve_column(df, user_id_col, USER_ID_CANDIDATES, "user")
    users = _series_values(df, user_id_col)
    dummy = [1.0] * len(users)

    if hue is None:
        labels = ["All Trips"] * len(users)
    elif hue == "day_of_week":
        day_col = _resolve_column(df, day_col, DAY_CANDIDATES, "day")
        labels = [None if _is_null(v) else str(v).lower() for v in _series_values(df, day_col)]
    elif hue == "day_period":
        if skip_day_period_creation and _DAY_PERIOD_COLUMN in df.columns:
            labels = _series_values(df, _DAY_PERIOD_COLUMN)
        else:
            trip_start_col = _resolve_column(df, trip_start_col, TIMESTAMP_CANDIDATES, "timestamp")
            labels = [_day_period(value) for value in _series_values(df, trip_start_col)]
    elif hue == "purpose":
        purpose_col = _resolve_column(df, purpose_col, ACTIVITY_CANDIDATES, "purpose")
        labels = _series_values(df, purpose_col)
    else:
        raise ValueError(f"Unsupported hue value: {hue!r}")

    counters: dict[Any, Counter] = defaultdict(Counter)
    for label, user, _ in zip(labels, users, dummy):
        if _is_null(label) or _is_null(user):
            continue
        counters[label][user] += 1
    return {label: list(counts.values()) for label, counts in counters.items()}


def visits_per_user_jensen_shannon_divergence(
    df1: Any,
    df2: Any,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    user_id_col1: str | None = None,
    user_id_col2: str | None = None,
    bin_size: float = 1.0,
    trip_start_col: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped visits-per-user distributions with Jensen-Shannon divergence."""
    groups1 = _visits_per_user_groups(
        df1,
        user_id_col=user_id_col1,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _visits_per_user_groups(
        df2,
        user_id_col=user_id_col2,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, histogram_jensen_shannon_divergence(groups1[label], groups2[label], bin_size=bin_size))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def visits_per_user_wasserstein_distance(
    df1: Any,
    df2: Any,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    user_id_col1: str | None = None,
    user_id_col2: str | None = None,
    trip_start_col: str | None = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
    skip_day_period_creation: bool = False,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare grouped visits-per-user distributions with Rust-backed Wasserstein distance."""
    groups1 = _visits_per_user_groups(
        df1,
        user_id_col=user_id_col1,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col1,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    groups2 = _visits_per_user_groups(
        df2,
        user_id_col=user_id_col2,
        hue=hue,
        trip_start_col=trip_start_col,
        day_col=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=skip_day_period_creation,
    )
    results = [
        (label, wasserstein_distance(groups1[label], groups2[label]))
        for label in _group_labels(groups1, groups2, hue)
        if label in groups1 and label in groups2
    ]
    return (float(np.mean([value for _, value in results])), results) if results else (float("nan"), [])


def od_matrix_common_part_of_commuters(od1: Any, od2: Any) -> float:
    """Return CPC between two OD matrices, aligning labels when available."""
    if all(hasattr(obj, attr) for obj in (od1, od2) for attr in ("index", "columns", "reindex")):
        origins = od1.index.union(od2.index)
        destinations = od1.columns.union(od2.columns)
        values1 = od1.reindex(index=origins, columns=destinations, fill_value=0).values
        values2 = od2.reindex(index=origins, columns=destinations, fill_value=0).values
    else:
        values1 = np.asarray(od1, dtype=np.float64)
        values2 = np.asarray(od2, dtype=np.float64)
        if values1.shape != values2.shape:
            raise ValueError(f"OD matrix shapes must match when labels are unavailable. Got {values1.shape} and {values2.shape}.")
    min_flows = float(np.minimum(values1, values2).sum())
    total_flows = float(np.asarray(values1).sum() + np.asarray(values2).sum())
    return 0.0 if total_flows == 0.0 else float(2.0 * min_flows / total_flows)


def activity_distribution_jensen_shannon_divergence(df1: Any, df2: Any, activity_col: str | None = None) -> float:
    """Compare categorical activity distributions with Jensen-Shannon divergence."""
    n1 = nw.from_native(df1, eager_only=True)
    n2 = nw.from_native(df2, eager_only=True)
    activity_col = activity_col or _pick_existing_column(n1.columns, ACTIVITY_CANDIDATES)
    if activity_col is None or activity_col not in n1.columns or activity_col not in n2.columns:
        raise ValueError(f"Could not detect activity column in both dataframes. Tried: {ACTIVITY_CANDIDATES}.")
    vals1 = [v for v in _series_values(n1, activity_col) if not _is_null(v)]
    vals2 = [v for v in _series_values(n2, activity_col) if not _is_null(v)]
    c1 = Counter(vals1)
    c2 = Counter(vals2)
    labels = sorted(set(c1) | set(c2), key=lambda value: str(value))
    return jensen_shannon_divergence([c1[label] for label in labels], [c2[label] for label in labels])


def activity_transition_matrix_jensen_shannon_divergence(
    matrix1: Any,
    matrix2: Any,
    categories1: list[Any] | None = None,
    categories2: list[Any] | None = None,
) -> float:
    """Compare activity transition matrices with Jensen-Shannon divergence."""
    return matrix_jensen_shannon_divergence(matrix1, matrix2, categories1, categories2)


def motif_distribution_jensen_shannon_divergence(
    visits1: Any,
    visits2: Any,
    *,
    user_id_col1: str | None = None,
    user_id_col2: str | None = None,
    location_id_col: str | None = None,
) -> float:
    """Discover daily motifs for two visit datasets and compare motif distributions."""
    daily1, _ = discover_daily_motifs_from_agents(visits1, user_id_col=user_id_col1, location_id_col=location_id_col)
    daily2, _ = discover_daily_motifs_from_agents(visits2, user_id_col=user_id_col2, location_id_col=location_id_col)
    n1 = nw.from_native(daily1, eager_only=True)
    n2 = nw.from_native(daily2, eager_only=True)
    motifs1 = _series_values(n1, "motif_id")
    motifs2 = _series_values(n2, "motif_id")
    c1 = Counter(motifs1)
    c2 = Counter(motifs2)
    labels = sorted(set(c1) | set(c2), key=lambda value: str(value))
    return jensen_shannon_divergence([c1[label] for label in labels], [c2[label] for label in labels])


def profile_metric_wasserstein_distance(df1: Any, df2: Any, metric_col: str) -> float:
    """Compare a scalar profile metric column with Rust-backed Wasserstein distance."""
    n1 = nw.from_native(df1, eager_only=True)
    n2 = nw.from_native(df2, eager_only=True)
    if metric_col not in n1.columns or metric_col not in n2.columns:
        raise ValueError(f"Column {metric_col!r} must be present in both dataframes.")
    return wasserstein_distance(n1.get_column(metric_col).to_numpy(), n2.get_column(metric_col).to_numpy())


def radius_of_gyration_wasserstein_distance(
    df1: Any,
    df2: Any,
    radius_col: str = "radius_of_gyration_km",
    grouping_col: str | None = None,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare radius-of-gyration distributions, optionally grouped by a column."""
    n1 = nw.from_native(df1, eager_only=True)
    if grouping_col is None or grouping_col not in n1.columns:
        value = profile_metric_wasserstein_distance(df1, df2, radius_col)
        return value, [("Overall", value)] if not np.isnan(value) else []
    return column_distribution_wasserstein_distance(df1, df2, radius_col, hue="purpose", purpose_col=grouping_col)


def dwell_time_wasserstein_distance(
    df1: Any,
    df2: Any,
    duration_col: str | None = None,
    *,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    day_col1: str | None = None,
    day_col2: str | None = None,
    purpose_col: str | None = None,
) -> tuple[float, list[tuple[Any, float]]]:
    """Compare dwell-time distributions in hours."""
    n1 = nw.from_native(df1, eager_only=True)
    duration_col = duration_col or _pick_existing_column(n1.columns, DURATION_CANDIDATES) or "duration_minutes"

    def add_hours(data: Any) -> Any:
        native = nw.from_native(data, eager_only=True)
        return native.with_columns((nw.col(duration_col) / 60.0).alias("__skmob2_dwell_hours__")).to_native()

    return column_distribution_wasserstein_distance(
        add_hours(df1),
        add_hours(df2),
        "__skmob2_dwell_hours__",
        hue=hue,
        day_col1=day_col1,
        day_col2=day_col2,
        purpose_col=purpose_col,
        skip_day_period_creation=True,
    )


def _wasserstein_column_wrapper(metric_col: str):
    def wrapper(df1: Any, df2: Any, metric_column: str = metric_col) -> float:
        return profile_metric_wasserstein_distance(df1, df2, metric_column)

    return wrapper


def compare_activity_distributions(visitation_df1: Any, visitation_df2: Any, act_column: str | None = None) -> float:
    return activity_distribution_jensen_shannon_divergence(visitation_df1, visitation_df2, activity_col=act_column)


compare_activity_10min_matrix = time_bin_matrix_jensen_shannon_divergence
compare_activity_transition_matrix = activity_transition_matrix_jensen_shannon_divergence


def compare_motif_distributions_with_jsd(
    agent_visitation_df: Any,
    sample_visitation_df: Any,
    user_id_col_agent: str | None = "agent_id",
    user_id_col_sample: str | None = "user_id",
    location_id_col: str | None = "area",
) -> float:
    return motif_distribution_jensen_shannon_divergence(
        agent_visitation_df,
        sample_visitation_df,
        user_id_col1=user_id_col_agent,
        user_id_col2=user_id_col_sample,
        location_id_col=location_id_col,
    )


compare_regularity_with_wasserstein = _wasserstein_column_wrapper("regularity")
compare_stationarity_with_wasserstein = _wasserstein_column_wrapper("stationarity")
compare_diversity_with_wasserstein = _wasserstein_column_wrapper("diversity")
compare_entropy_with_wasserstein = _wasserstein_column_wrapper("entropy")
compare_predictability_with_wasserstein = _wasserstein_column_wrapper("predictability")
compute_cpc_of_od_matrix = od_matrix_common_part_of_commuters


def compare_distributions_with_js_divergence(
    df1: Any,
    df2: Any,
    column_to_compare: str,
    bin_size: float = 1.0,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_column: str | None = "Time_O",
    day_column1: str | None = "Day_EMG",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "Purpose_D",
    skip_day_period_creation: bool = False,
):
    return column_distribution_jensen_shannon_divergence(
        df1,
        df2,
        column_to_compare,
        bin_size=bin_size,
        hue=hue,
        trip_start_col1=trip_start_column,
        trip_start_col2=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_visits_per_user_with_js_divergence(
    df1: Any,
    df2: Any,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    user_id_column1: str | None = "user_id",
    user_id_column2: str | None = "user_id",
    bin_size: float = 1.0,
    skip_day_period_creation: bool = False,
    trip_start_column: str | None = "start_timestamp",
    day_column1: str | None = "Day_EMG",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "purpose",
):
    return visits_per_user_jensen_shannon_divergence(
        df1,
        df2,
        hue=hue,
        user_id_col1=user_id_column1,
        user_id_col2=user_id_column2,
        bin_size=bin_size,
        trip_start_col=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_distributions_with_wasserstein(
    df1: Any,
    df2: Any,
    column_to_compare: str,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_column: str | None = "Time_O",
    day_column1: str | None = "Day_EMG",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "Purpose_D",
    skip_day_period_creation: bool = False,
):
    return column_distribution_wasserstein_distance(
        df1,
        df2,
        column_to_compare,
        hue=hue,
        trip_start_col1=trip_start_column,
        trip_start_col2=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_visits_per_user_with_wasserstein(
    df1: Any,
    df2: Any,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    user_id_column1: str | None = "user_id",
    user_id_column2: str | None = "user_id",
    trip_start_column: str | None = "start_timestamp",
    day_column1: str | None = "day_of_week",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "purpose",
    skip_day_period_creation: bool = False,
):
    return visits_per_user_wasserstein_distance(
        df1,
        df2,
        hue=hue,
        user_id_col1=user_id_column1,
        user_id_col2=user_id_column2,
        trip_start_col=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_trip_length_with_wasserstein(
    trips_df1: Any,
    trips_df2: Any,
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_column: str | None = "Time_O",
    day_column1: str | None = "Day_EMG",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "Purpose_D",
):
    kwargs = {
        "hue": hue,
        "trip_start_column": trip_start_column,
        "day_column1": day_column1,
        "day_column2": day_column2,
        "purpose_column": purpose_column,
    }
    return compare_distributions_with_wasserstein(trips_df1, trips_df2, "trip_length_km", **kwargs)


def compare_trip_duration_with_wasserstein(
    trips_df1: Any,
    trips_df2: Any,
    duration_column: str = "Duration",
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    trip_start_column: str | None = "Time_O",
    day_column1: str | None = "Day_EMG",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "Purpose_D",
):
    kwargs = {
        "hue": hue,
        "trip_start_column": trip_start_column,
        "day_column1": day_column1,
        "day_column2": day_column2,
        "purpose_column": purpose_column,
    }
    return compare_distributions_with_wasserstein(trips_df1, trips_df2, duration_column, **kwargs)


def compare_dwell_time_with_wasserstein(
    visitation_df1: Any,
    visitation_df2: Any,
    duration_column: str = "duration_minutes",
    hue: Literal[None, "day_of_week", "day_period", "purpose"] = None,
    day_column1: str | None = "day_of_week",
    day_column2: str | None = "day_of_week",
    purpose_column: str | None = "purpose",
):
    return dwell_time_wasserstein_distance(
        visitation_df1,
        visitation_df2,
        duration_col=duration_column,
        hue=hue,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
    )


def compare_radius_of_gyration_with_wasserstein(
    rg_df1: Any,
    rg_df2: Any,
    radius_column: str = "radius_of_gyration_km",
    grouping_column: str | None = None,
):
    return radius_of_gyration_wasserstein_distance(
        rg_df1,
        rg_df2,
        radius_col=radius_column,
        grouping_col=grouping_column,
    )

__all__ = [
    "jensen_shannon_divergence",
    "matrix_jensen_shannon_divergence",
    "time_bin_matrix_jensen_shannon_divergence",
    "histogram_jensen_shannon_divergence",
    "wasserstein_distance",
    "column_distribution_jensen_shannon_divergence",
    "column_distribution_wasserstein_distance",
    "visits_per_user_jensen_shannon_divergence",
    "visits_per_user_wasserstein_distance",
    "od_matrix_common_part_of_commuters",
    "activity_distribution_jensen_shannon_divergence",
    "activity_transition_matrix_jensen_shannon_divergence",
    "motif_distribution_jensen_shannon_divergence",
    "profile_metric_wasserstein_distance",
    "radius_of_gyration_wasserstein_distance",
    "dwell_time_wasserstein_distance",
    "compare_activity_distributions",
    "compare_activity_10min_matrix",
    "compare_activity_transition_matrix",
    "compare_motif_distributions_with_jsd",
    "compare_regularity_with_wasserstein",
    "compare_stationarity_with_wasserstein",
    "compare_diversity_with_wasserstein",
    "compare_entropy_with_wasserstein",
    "compare_predictability_with_wasserstein",
    "compare_distributions_with_js_divergence",
    "compare_visits_per_user_with_js_divergence",
    "compute_cpc_of_od_matrix",
    "compare_distributions_with_wasserstein",
    "compare_trip_length_with_wasserstein",
    "compare_trip_duration_with_wasserstein",
    "compare_dwell_time_with_wasserstein",
    "compare_radius_of_gyration_with_wasserstein",
    "compare_visits_per_user_with_wasserstein",
]
