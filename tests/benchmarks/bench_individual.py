"""Benchmarks for skmob2/measures/individual.py.

Requires pytest-benchmark. Run with:
    pytest tests/benchmarks/ -v
    pytest tests/benchmarks/ --benchmark-json=results.json
    pytest tests/benchmarks/ --benchmark-save=baseline
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# jump_lengths
# ---------------------------------------------------------------------------


def test_jump_lengths_skmob2_pandas(benchmark, bench_slice_pandas):
    """Benchmark skmob2.jump_lengths on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.individual import jump_lengths as skmob2_jl

    benchmark(skmob2_jl, bench_slice_pandas, show_progress=False, merge=False)


def test_jump_lengths_skmob(benchmark, bench_slice_skmob):
    """Benchmark skmob.jump_lengths on a TrajDataFrame."""
    skmob_individual = pytest.importorskip(
        "skmob.measures.individual",
        reason="Install skmob to run this benchmark",
    )
    skmob_jl = skmob_individual.jump_lengths
    benchmark(skmob_jl, bench_slice_skmob, show_progress=False, merge=False)


def test_jump_lengths_skmob2_polars(benchmark, bench_slice_polars):
    """Benchmark skmob2.jump_lengths on a Polars DataFrame (dataframe-agnostic test)."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.individual import jump_lengths as skmob2_jl

    benchmark(skmob2_jl, bench_slice_polars, show_progress=False, merge=False)


# ---------------------------------------------------------------------------
# radius_of_gyration
# ---------------------------------------------------------------------------


def test_radius_of_gyration_skmob2_pandas(benchmark, bench_slice_pandas):
    """Benchmark skmob2.radius_of_gyration on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.individual import radius_of_gyration as skmob2_rog

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
    from skmob2.measures.individual import radius_of_gyration as skmob2_rog

    benchmark(skmob2_rog, bench_slice_polars, show_progress=False)
