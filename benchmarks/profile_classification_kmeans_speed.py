"""Compare sklearn and FastMOB's Arrow-native profile-classification K-Means.

The sklearn implementation is benchmark-only. FastMOB standardizes the two
profile features and performs one deterministic K-Means++ initialisation in
Rust, which is the production profile-classification configuration.
Run with ``python benchmarks/profile_classification_kmeans_speed.py`` from the
development environment to report median latency and Python allocation peak.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np
import pyarrow as pa

from fastmob._core import cluster_standardized_kmeans_arrow


def legacy_sklearn(first: np.ndarray, second: np.ndarray):
    """The former StandardScaler + sklearn KMeans implementation."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = np.column_stack([first, second])
    return KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(features))


def native_rust(first: pa.Array, second: pa.Array):
    return cluster_standardized_kmeans_arrow(first, second, n_clusters=3, seed=0)


def measure(func, args: tuple[object, ...], repeats: int) -> tuple[float, float]:
    durations, peaks_mb = [], []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        labels = func(*args)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(labels) == len(args[0])
        durations.append(elapsed)
        peaks_mb.append(peak / (1024 * 1024))
    return statistics.median(durations), statistics.mean(peaks_mb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    for n_users in args.users:
        # Three well-separated groups reflect the Routiner/Regular/Scouter
        # workload while retaining realistic within-profile variation.
        group = rng.integers(0, 3, n_users)
        first = rng.normal(np.take([1.0, 3.0, 6.0], group), 0.2).astype(np.float64)
        second = rng.normal(np.take([0.9, 0.55, 0.15], group), 0.04).astype(np.float64)
        arrow_values = (pa.array(first), pa.array(second))

        legacy_sklearn(first, second)
        native_rust(*arrow_values)
        legacy_s, legacy_mb = measure(legacy_sklearn, (first, second), args.repeats)
        native_s, native_mb = measure(native_rust, arrow_values, args.repeats)
        print(
            f"users={n_users:,}: legacy={legacy_s:.4f}s/{legacy_mb:.2f}MiB, "
            f"native={native_s:.4f}s/{native_mb:.2f}MiB, "
            f"speedup={legacy_s / native_s:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
