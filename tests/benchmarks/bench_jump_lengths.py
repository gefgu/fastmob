"""Benchmarks for skmob2/measures/jump_lengths.py."""
from __future__ import annotations

import pytest


def test_jump_lengths_skmob2_pandas(benchmark, bench_slice_pandas):
    """Benchmark skmob2.jump_lengths on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.jump_lengths import jump_lengths as skmob2_jl

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
    """Benchmark skmob2.jump_lengths on a Polars DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.jump_lengths import jump_lengths as skmob2_jl

    benchmark(skmob2_jl, bench_slice_polars, show_progress=False, merge=False)
