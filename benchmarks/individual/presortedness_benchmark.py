"""Measure Rust presortedness checks and indexed/presorted individual paths.

Example::

    python benchmarks/individual/presortedness_benchmark.py \
        --data-path tests/shared/data/loc-brightkite_totalCheckins.txt.gz \
        --backend both --sizes 10000,100000,1000000

The benchmark deliberately times checks separately from dataframe setup and
sorting.  This makes the result useful for deciding whether automatic
dispatch to the existing contiguous kernels is worthwhile.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from benchmarks.individual.speed_suite import (
    DEFAULT_DATA_PATH,
    load_brightkite_pandas,
    load_brightkite_polars,
)


def _clean(df: Any, backend: str) -> Any:
    required = ["user", "check-in_time", "latitude", "longitude"]
    if backend == "polars":
        import polars as pl

        return df.drop_nulls(subset=required).filter(
            pl.col("latitude").is_not_nan() & pl.col("longitude").is_not_nan()
        )
    return df.dropna(subset=required).reset_index(drop=True)


def _sort(df: Any, backend: str, columns: list[str]) -> Any:
    if backend == "polars":
        return df.sort(columns, maintain_order=True)
    return df.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _arrow_column(df: Any, name: str) -> pa.Array:
    return pa.array(df.get_column(name).to_arrow() if hasattr(df, "get_column") else df[name].array)


def _numpy_timestamps(df: Any) -> np.ndarray:
    values = _arrow_column(df, "check-in_time")
    return np.asarray(values.to_numpy(zero_copy_only=False))


def _rust_run_boundaries(df: Any) -> tuple[pa.Array, pa.Array] | None:
    """``(run leaders, run ends)`` via the Rust scan, or None for a declined dtype."""
    from fastmob._core import value_run_boundaries

    users = _arrow_column(df, "user")
    boundaries = value_run_boundaries(users)
    if boundaries is None:
        return None
    starts, ends = boundaries
    return pc.take(users, pa.array(starts)), pa.array(ends)


def check_users_contiguous_rust(df: Any) -> bool:
    """Contiguity exactly as the measures now establish it: no uid leads two runs."""
    boundaries = _rust_run_boundaries(df)
    if boundaries is None:
        return check_users_contiguous(df)
    leaders, _ = boundaries
    return len(pc.unique(leaders)) == len(leaders)


def check_users_and_timestamps_sorted_rust(df: Any) -> bool:
    """Contiguity plus within-run time order, as ``jump_lengths`` establishes it."""
    from fastmob._core import validate_timestamps_within_ends

    boundaries = _rust_run_boundaries(df)
    if boundaries is None:
        return check_users_and_timestamps_sorted(df)
    leaders, ends = boundaries
    if len(pc.unique(leaders)) != len(leaders):
        return False
    return bool(validate_timestamps_within_ends(_arrow_column(df, "check-in_time"), ends))


def check_users_contiguous(df: Any) -> bool:
    """Return whether every user occupies exactly one contiguous run."""
    users = _arrow_column(df, "user")
    if len(users) < 2:
        return True
    runs = pc.run_end_encode(users)
    return len(pc.unique(runs.values)) == len(runs.values)


def check_users_and_timestamps_sorted(df: Any) -> bool:
    """Return whether users are contiguous and timestamps rise within runs."""
    users = _arrow_column(df, "user")
    if not check_users_contiguous(df) or len(users) < 2:
        return check_users_contiguous(df)

    timestamps = _numpy_timestamps(df)
    same_user = np.asarray(
        pc.equal(users.slice(1), users.slice(0, len(users) - 1)), dtype=bool
    )
    return bool(np.all(~same_user | (timestamps[1:] >= timestamps[:-1])))


def _time_call(func: Callable[[], Any], iterations: int) -> dict[str, Any]:
    func()
    times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        func()
        times.append(time.perf_counter() - started)
    return {
        "status": "ok",
        "times_seconds": times,
        "iterations_completed": len(times),
        "average_seconds": sum(times) / len(times),
        "minimum_seconds": min(times),
    }


def _metric_call(name: str, df: Any) -> Any:
    from fastmob.measures.individual.jump_lengths import jump_lengths
    from fastmob.measures.individual.radius_of_gyration import radius_of_gyration

    kwargs = {
        "uid_col": "user",
        "datetime_col": "check-in_time",
        "lat_col": "latitude",
        "lng_col": "longitude",
    }
    if name == "jump_lengths":
        return jump_lengths(df, merge=True, **kwargs)
    return radius_of_gyration(df, **kwargs)


def _kernel_only_checks(df: Any) -> tuple[Callable[[], bool], Callable[[], bool]]:
    """Bind the dispatch decision to already-extracted Arrow columns.

    ``check_users_*_rust`` re-extract the uid and timestamp columns on every
    call, which at 4M rows costs far more than the decision itself -- the
    harness would otherwise report mostly ``pa.array``.  The measures hold these
    buffers already, so the kernel-only number is what their dispatch adds.
    """
    from fastmob._core import validate_timestamps_within_ends, value_run_boundaries

    users = _arrow_column(df, "user")
    stamps = _arrow_column(df, "check-in_time")

    def grouped() -> bool:
        starts, _ends = value_run_boundaries(users)
        leaders = pc.take(users, pa.array(starts))
        return len(pc.unique(leaders)) == len(leaders)

    def grouped_and_time_ordered() -> bool:
        starts, ends = value_run_boundaries(users)
        leaders = pc.take(users, pa.array(starts))
        if len(pc.unique(leaders)) != len(leaders):
            return False
        return bool(validate_timestamps_within_ends(stamps, pa.array(ends)))

    return grouped, grouped_and_time_ordered


def _timed_metrics(df: Any, backend: str, order: str, iterations: int) -> dict[str, Any]:
    """Time the dispatch decision and the measures that now make it themselves.

    ``jump_lengths`` and ``radius_of_gyration`` detect an already-grouped frame
    and route to their contiguous kernels without being told, so there is no
    longer an indexed-vs-presorted pair to time here -- the interesting number
    is the same measure call across input orderings, which is what the ``order``
    dimension of this benchmark already varies.
    """
    user_kernel, jump_kernel = _kernel_only_checks(df)

    metrics = {
        "check_user_contiguous_rust": _time_call(
            lambda: check_users_contiguous_rust(df), iterations
        ),
        "check_user_timestamp_sorted_rust": _time_call(
            lambda: check_users_and_timestamps_sorted_rust(df), iterations
        ),
        "check_user_contiguous_kernel": _time_call(user_kernel, iterations),
        "check_user_timestamp_sorted_kernel": _time_call(jump_kernel, iterations),
        "jump_lengths": _time_call(lambda: _metric_call("jump_lengths", df), iterations),
        "radius_of_gyration": _time_call(
            lambda: _metric_call("radius_of_gyration", df), iterations
        ),
    }

    metrics["derived"] = {
        "jump_takes_contiguous_path": bool(jump_kernel()),
        "radius_takes_contiguous_path": bool(user_kernel()),
        "jump_dispatch_check_seconds": metrics["check_user_timestamp_sorted_kernel"][
            "average_seconds"
        ],
        "radius_dispatch_check_seconds": metrics["check_user_contiguous_kernel"][
            "average_seconds"
        ],
    }
    return metrics


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def run(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for backend in args.backends:
        loader = load_brightkite_pandas if backend == "pandas" else load_brightkite_polars
        raw_full = loader(args.data_path)
        for size in args.sizes:
            raw = _clean(raw_full.head(size), backend)
            cases = [
                ("raw", raw),
                ("user_sorted", _sort(raw, backend, ["user"])),
                ("user_timestamp_sorted", _sort(raw, backend, ["user", "check-in_time"])),
            ]
            for order, frame in cases:
                print(f"{backend} {order} {len(frame):,} rows")
                results.append(
                    {
                        "backend": backend,
                        "order": order,
                        "rows": len(frame),
                        "metrics": _timed_metrics(frame, backend, order, args.iterations),
                    }
                )
    payload = {
        "metadata": {
            "dataset_path": str(args.data_path),
            "backends": args.backends,
            "sizes": args.sizes,
            "iterations": args.iterations,
            "ordering_rule": "contiguous user groups; timestamps nondecreasing within group",
            "validator": "fastmob._core.value_run_boundaries + validate_timestamps_within_ends",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--sizes", type=_parse_sizes, default=[1_000, 10_000, 100_000, 1_000_000, 4_000_000])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/presortedness_benchmark.json"))
    args = parser.parse_args()
    args.backends = ["pandas", "polars"] if args.backend == "both" else [args.backend]
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
