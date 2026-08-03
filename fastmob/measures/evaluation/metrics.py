"""Core Jensen-Shannon and Wasserstein primitives.

Numpy + simsimd for Jensen-Shannon; Wasserstein is Arrow-first into the Rust
kernel. No Narwhals dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fastmob._core import wasserstein as _wasserstein


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


def _finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


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


def _finite_arrow_array(values: Any) -> pa.Array:
    arr = pa.array(values, type=pa.float64())
    return pc.filter(arr, pc.is_finite(arr))


def wasserstein_distance(values1: Any, values2: Any) -> float:
    """Return Rust-backed 1D Wasserstein distance between empirical samples."""
    v1 = _finite_arrow_array(values1)
    v2 = _finite_arrow_array(values2)
    if len(v1) == 0 or len(v2) == 0:
        return float("nan")
    return float(_wasserstein(v1, v2))
