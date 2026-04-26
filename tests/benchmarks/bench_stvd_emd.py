"""Benchmarks for skmob2/measures/stvd_emd.py — zero-copy numpy vs arrow paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_dist_df(n: int, backend: str = "pandas"):
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 10000, n)
    ys = rng.uniform(0, 10000, n)
    hours = rng.integers(0, 24, n)
    minutes = rng.integers(0, 60, n)
    weights = rng.uniform(0.1, 1.0, n)
    data = {
        "centroid": [f"POINT ({x:.2f} {y:.2f})" for x, y in zip(xs, ys)],
        "time_bin": [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)],
        "mean_volume": weights,
    }
    if backend == "pandas":
        return pd.DataFrame(data)
    polars = pytest.importorskip("polars")
    return polars.DataFrame(data)


@pytest.mark.parametrize("n_cells", [50, 200, 1000])
def test_stvd_emd_pandas(benchmark, n_cells):
    """Benchmark stvd_emd on pandas DataFrames (numpy zero-copy path)."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.stvd_emd import stvd_emd

    df_a = _make_dist_df(n_cells, "pandas")
    df_b = _make_dist_df(n_cells, "pandas")
    benchmark(stvd_emd, df_a, df_b)


@pytest.mark.parametrize("n_cells", [50, 200, 1000])
def test_stvd_emd_polars(benchmark, n_cells):
    """Benchmark stvd_emd on polars DataFrames (Arrow zero-copy path)."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pytest.importorskip("polars", reason="Install polars to run this benchmark")
    from skmob2.measures.stvd_emd import stvd_emd

    df_a = _make_dist_df(n_cells, "polars")
    df_b = _make_dist_df(n_cells, "polars")
    benchmark(stvd_emd, df_a, df_b)
