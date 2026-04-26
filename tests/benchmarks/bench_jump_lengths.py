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


@pytest.mark.movingpandas
def test_jump_lengths_movingpandas_1k(benchmark, bench_slice_movingpandas_1k):
    """Benchmark movingpandas TrajectoryCollection.add_distance (km) on 1k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(bench_slice_movingpandas_1k.add_distance, overwrite=True, units="km")


@pytest.mark.movingpandas
def test_jump_lengths_movingpandas_10k(benchmark, bench_slice_movingpandas_10k):
    """Benchmark movingpandas TrajectoryCollection.add_distance (km) on 10k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(bench_slice_movingpandas_10k.add_distance, overwrite=True, units="km")


@pytest.mark.movingpandas
def test_jump_lengths_movingpandas_100k(benchmark, bench_slice_movingpandas_100k):
    """Benchmark movingpandas TrajectoryCollection.add_distance (km) on 100k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(bench_slice_movingpandas_100k.add_distance, overwrite=True, units="km")


@pytest.mark.movingpandas
def test_jump_lengths_movingpandas_1M(benchmark, bench_slice_movingpandas_1M):
    """Benchmark movingpandas TrajectoryCollection.add_distance (km) on 1M rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(bench_slice_movingpandas_1M.add_distance, overwrite=True, units="km")


@pytest.mark.movingpandas
def test_jump_lengths_movingpandas_4M(benchmark, bench_slice_movingpandas_4M):
    """Benchmark movingpandas TrajectoryCollection.add_distance (km) on 4M rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(bench_slice_movingpandas_4M.add_distance, overwrite=True, units="km")
