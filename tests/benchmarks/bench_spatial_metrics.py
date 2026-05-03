"""Benchmarks for spatial metrics without dedicated benchmark modules."""

from __future__ import annotations

import importlib

import pytest


_SPATIAL_METRICS = [
    ("distance_straight_line", "skmob2.measures.spatial.distance_straight_line", "distance_straight_line", {}),
    ("maximum_distance", "skmob2.measures.spatial.maximum_distance", "maximum_distance", {}),
    ("waiting_times", "skmob2.measures.spatial.waiting_times", "waiting_times", {"merge": False}),
    ("k_radius_of_gyration", "skmob2.measures.spatial.k_radius_of_gyration", "k_radius_of_gyration", {}),
    ("number_of_visits", "skmob2.measures.spatial.number_of_visits", "number_of_visits", {}),
    ("number_of_locations", "skmob2.measures.spatial.number_of_locations", "number_of_locations", {}),
    ("home_location", "skmob2.measures.spatial.home_location", "home_location", {}),
    ("max_distance_from_home", "skmob2.measures.spatial.max_distance_from_home", "max_distance_from_home", {}),
]


@pytest.mark.parametrize("metric_name,module_path,func_name,kwargs", _SPATIAL_METRICS, ids=[m[0] for m in _SPATIAL_METRICS])
def test_spatial_metric_skmob2_pandas(benchmark, bench_slice_pandas, metric_name, module_path, func_name, kwargs):
    """Benchmark skmob2 spatial metrics on a plain pandas DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    del metric_name
    func = getattr(importlib.import_module(module_path), func_name)
    benchmark(func, bench_slice_pandas, **kwargs)


@pytest.mark.parametrize("metric_name,module_path,func_name,kwargs", _SPATIAL_METRICS, ids=[m[0] for m in _SPATIAL_METRICS])
def test_spatial_metric_skmob2_polars(benchmark, bench_slice_polars, metric_name, module_path, func_name, kwargs):
    """Benchmark skmob2 spatial metrics on a Polars DataFrame."""
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )
    del metric_name
    func = getattr(importlib.import_module(module_path), func_name)
    benchmark(func, bench_slice_polars, **kwargs)
