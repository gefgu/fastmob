"""Core Jensen-Shannon and Wasserstein primitives.

Both metrics are Arrow-in, Rust-out: inputs are converted to Arrow float64
arrays in Python and the numeric work happens in the ``jensen_shannon`` and
``wasserstein`` Rust kernels. No Narwhals or numpy dependency.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from fastmob._core import jensen_shannon as _jensen_shannon_divergence
from fastmob._core import wasserstein as _wasserstein
from fastmob.utils._common import _finite_arrow_array


def jensen_shannon_divergence(distribution1: Any, distribution2: Any) -> float:
    """Return Jensen-Shannon divergence between two distributions.

    Inputs are normalised internally (raw counts and probabilities both
    work) and must have the same length -- unlike :func:`wasserstein_distance`,
    the two arrays are paired by index (one entry per category/bin), not
    independent samples.
    """
    p = pa.array(distribution1, type=pa.float64())
    q = pa.array(distribution2, type=pa.float64())
    return float(_jensen_shannon_divergence(p, q))


def wasserstein_distance(values1: Any, values2: Any) -> float:
    """Return Rust-backed 1D Wasserstein distance between empirical samples."""
    v1 = _finite_arrow_array(values1)
    v2 = _finite_arrow_array(values2)
    if len(v1) == 0 or len(v2) == 0:
        return float("nan")
    return float(_wasserstein(v1, v2))
