"""Benchmarks for skmob2/measures/spatial/radius_of_gyration.py."""

from __future__ import annotations

import numpy as np
import pytest


def _rog_for_tc(tc):
    """Compute radius of gyration for every trajectory in a TrajectoryCollection.

    For each trajectory, extracts (x, y) coordinates from the GeoDataFrame geometry,
    computes the centroid, and returns the RMS distance from each point to the centroid.

    Parameters:
        tc: movingpandas.TrajectoryCollection whose trajectories are iterated.

    Returns:
        list[float]: one RoG value per trajectory.

    @usedBy: test_radius_of_gyration_movingpandas_* benchmark functions in this file.
    """
    results = []
    for traj in tc.trajectories:
        coords = np.array([[p.x, p.y] for p in traj.df.geometry])
        center = coords.mean(axis=0)
        rog = np.sqrt(((coords - center) ** 2).sum(axis=1).mean())
        results.append(rog)
    return results


def test_radius_of_gyration_skmob2_pandas(benchmark, bench_slice_pandas):
    """Benchmark skmob2.radius_of_gyration on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as skmob2_rog

    benchmark(skmob2_rog, bench_slice_pandas)


@pytest.mark.skmob
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
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as skmob2_rog

    benchmark(skmob2_rog, bench_slice_polars)


@pytest.mark.movingpandas
def test_radius_of_gyration_movingpandas_1k(benchmark, bench_slice_movingpandas_1k):
    """Benchmark hand-rolled NumPy radius-of-gyration over a movingpandas TrajectoryCollection on 1k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(_rog_for_tc, bench_slice_movingpandas_1k)


@pytest.mark.movingpandas
def test_radius_of_gyration_movingpandas_10k(benchmark, bench_slice_movingpandas_10k):
    """Benchmark hand-rolled NumPy radius-of-gyration over a movingpandas TrajectoryCollection on 10k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(_rog_for_tc, bench_slice_movingpandas_10k)


@pytest.mark.movingpandas
def test_radius_of_gyration_movingpandas_100k(benchmark, bench_slice_movingpandas_100k):
    """Benchmark hand-rolled NumPy radius-of-gyration over a movingpandas TrajectoryCollection on 100k rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(_rog_for_tc, bench_slice_movingpandas_100k)


@pytest.mark.movingpandas
def test_radius_of_gyration_movingpandas_1M(benchmark, bench_slice_movingpandas_1M):
    """Benchmark hand-rolled NumPy radius-of-gyration over a movingpandas TrajectoryCollection on 1M rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(_rog_for_tc, bench_slice_movingpandas_1M)


@pytest.mark.movingpandas
def test_radius_of_gyration_movingpandas_4M(benchmark, bench_slice_movingpandas_4M):
    """Benchmark hand-rolled NumPy radius-of-gyration over a movingpandas TrajectoryCollection on 4M rows."""
    pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    benchmark(_rog_for_tc, bench_slice_movingpandas_4M)
