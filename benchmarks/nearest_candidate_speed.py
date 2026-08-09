"""Compare the legacy scikit-learn and native Rust nearest-candidate paths.

The scikit-learn path is benchmark-only: production code has no dependency on
it.  Run with ``python benchmarks/nearest_candidate_speed.py`` from the dev
environment to obtain median latency and Python allocation peak for both
implementations.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
import tracemalloc

import numpy as np

from fastmob._core import haversine_m_batch
from fastmob.network._nearest import nearest_candidate


def legacy_sklearn(query_lat, query_lng, ref_lat, ref_lng):
    """The pre-native nearest-candidate algorithm, retained only as a baseline."""
    from sklearn.neighbors import NearestNeighbors

    scale = math.cos(math.radians(float(np.mean(ref_lat))))
    tree = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    tree.fit(np.column_stack([ref_lat, ref_lng * scale]))
    _, indices = tree.kneighbors(np.column_stack([query_lat, query_lng * scale]))
    nearest_idx = indices[:, 0]
    distances = haversine_m_batch(query_lat, query_lng, ref_lat[nearest_idx], ref_lng[nearest_idx])
    return nearest_idx, distances


def measure(func, args, repeats: int):
    samples = []
    peaks_mb = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        indices, distances = func(*args)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(indices) == len(args[0]) == len(distances)
        samples.append(elapsed)
        peaks_mb.append(peak / (1024 * 1024))
    return statistics.median(samples), statistics.mean(peaks_mb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=100_000)
    parser.add_argument("--references", type=int, nargs="+", default=[500, 10_000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    query_lat = rng.uniform(40.0, 40.1, args.queries)
    query_lng = rng.uniform(-74.0, -73.9, args.queries)
    for n_references in args.references:
        ref_lat = rng.uniform(40.0, 40.1, n_references)
        ref_lng = rng.uniform(-74.0, -73.9, n_references)
        values = (query_lat, query_lng, ref_lat, ref_lng)
        legacy_sklearn(*values)  # warm-up imports and native allocations
        nearest_candidate(*values)
        legacy_s, legacy_mb = measure(legacy_sklearn, values, args.repeats)
        native_s, native_mb = measure(nearest_candidate, values, args.repeats)
        print(
            f"queries={args.queries:,} references={n_references:,}: "
            f"legacy={legacy_s:.4f}s/{legacy_mb:.2f}MiB, "
            f"native={native_s:.4f}s/{native_mb:.2f}MiB, "
            f"speedup={legacy_s / native_s:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
