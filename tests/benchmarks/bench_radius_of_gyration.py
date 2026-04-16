"""Benchmarks for skmob2/measures/radius_of_gyration.py."""
from __future__ import annotations

import pytest


def test_radius_of_gyration_skmob2_pandas(benchmark, bench_slice_pandas):
    """Benchmark skmob2.radius_of_gyration on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.radius_of_gyration import radius_of_gyration as skmob2_rog

    benchmark(skmob2_rog, bench_slice_pandas, show_progress=False)


def test_radius_of_gyration_skmob(benchmark, bench_slice_skmob):
    """Benchmark skmob.radius_of_gyration on a TrajDataFrame."""
    skmob_individual = pytest.importorskip(
        "skmob.measures.individual",
        reason="Install skmob to run this benchmark",
    )
    skmob_rog = skmob_individual.radius_of_gyration
    benchmark(skmob_rog, bench_slice_skmob, show_progress=False)


def test_radius_of_gyration_skmob2_polars(benchmark, bench_slice_polars):
    """Benchmark skmob2.radius_of_gyration on a Polars DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.radius_of_gyration import radius_of_gyration as skmob2_rog

    benchmark(skmob2_rog, bench_slice_polars, show_progress=False)
