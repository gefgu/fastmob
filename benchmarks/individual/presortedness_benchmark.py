"""Measure Rust presortedness checks and indexed/presorted individual paths.

Example::

    python benchmarks/individual/presortedness_benchmark.py \
        --data-path tests/shared/data/loc-brightkite_totalCheckins.txt.gz \
        --backend both --sizes 10000,100000,1000000

The benchmark deliberately times checks separately from dataframe setup and
sorting.  This makes the result useful for deciding whether automatic
dispatch to the existing ``presorted=True`` kernels is worthwhile.
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


def _rust_user_codes(df: Any) -> pa.Array:
    users = _arrow_column(df, "user")
    try:
        return users.cast(pa.uint32())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        from fastmob.utils._common import _factorize_arrow_values

        codes, _ = _factorize_arrow_values(users, sort=False)
        return pa.array(codes, type=pa.uint32())


def check_users_contiguous_rust(df: Any) -> bool:
    from fastmob._core import validate_presorted_user_timestamps

    return bool(validate_presorted_user_timestamps(_rust_user_codes(df), None, check_timestamps=False))


def check_users_and_timestamps_sorted_rust(df: Any) -> bool:
    from fastmob._core import validate_presorted_user_timestamps

    return bool(
        validate_presorted_user_timestamps(
            _rust_user_codes(df),
            _arrow_column(df, "check-in_time"),
            check_timestamps=True,
        )
    )


def check_users_contiguous_rust_new(df: Any) -> bool:
    from fastmob._core import validate_grouped_user_timestamps

    return bool(validate_grouped_user_timestamps(_rust_user_codes(df), check_timestamps=False))


def check_users_and_timestamps_sorted_rust_new(df: Any) -> bool:
    from fastmob._core import validate_grouped_user_timestamps

    return bool(
        validate_grouped_user_timestamps(
            _rust_user_codes(df),
            _arrow_column(df, "check-in_time"),
            check_timestamps=True,
        )
    )


def check_users_contiguous_rust(df: Any) -> bool:
    """Compatibility alias for the current grouped Rust validator."""
    return check_users_contiguous_rust_new(df)


def check_users_and_timestamps_sorted_rust(df: Any) -> bool:
    """Compatibility alias for the current grouped Rust validator."""
    return check_users_and_timestamps_sorted_rust_new(df)


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


def _metric_call(name: str, df: Any, *, presorted: bool) -> Any:
    from fastmob.measures.individual.jump_lengths import jump_lengths
    from fastmob.measures.individual.radius_of_gyration import radius_of_gyration

    kwargs = {
        "uid_col": "user",
        "datetime_col": "check-in_time",
        "lat_col": "latitude",
        "lng_col": "longitude",
        "presorted": presorted,
    }
    if name == "jump_lengths":
        return jump_lengths(df, merge=True, **kwargs)
    return radius_of_gyration(df, **kwargs)


def _kernel_only_checks(df: Any) -> tuple[Callable[[], bool], Callable[[], bool]]:
    """Bind the validator to already-extracted Arrow columns.

    ``check_users_*_rust_new`` re-extract and re-cast the uid and timestamp
    columns on every call, which at 4M rows costs 15-22 ms against a validator
    that costs well under 1 ms -- the harness would otherwise report almost
    nothing but ``pa.array``/``cast``.  Real measure wrappers already hold these
    buffers, so the kernel-only number is the one that says what a presortedness
    check actually adds.
    """
    from fastmob._core import validate_grouped_user_timestamps

    codes = _rust_user_codes(df)
    stamps = _arrow_column(df, "check-in_time")
    return (
        lambda: bool(validate_grouped_user_timestamps(codes, check_timestamps=False)),
        lambda: bool(validate_grouped_user_timestamps(codes, stamps, check_timestamps=True)),
    )


def _timed_metrics(df: Any, backend: str, order: str, iterations: int) -> dict[str, Any]:
    jump_check = check_users_and_timestamps_sorted_rust_new
    user_check = check_users_contiguous_rust_new
    user_kernel, jump_kernel = _kernel_only_checks(df)

    jump_sorted = order == "user_timestamp_sorted"
    user_sorted = order in {"user_sorted", "user_timestamp_sorted"}

    def auto_jump() -> Any:
        return _metric_call("jump_lengths", df, presorted=jump_check(df))

    def auto_radius() -> Any:
        return _metric_call("radius_of_gyration", df, presorted=user_check(df))

    def checked_indexed_jump() -> Any:
        jump_check(df)
        return _metric_call("jump_lengths", df, presorted=False)

    def checked_indexed_radius() -> Any:
        user_check(df)
        return _metric_call("radius_of_gyration", df, presorted=False)

    metrics = {
        "check_user_contiguous_rust": _time_call(lambda: user_check(df), iterations),
        "check_user_timestamp_sorted_rust": _time_call(lambda: jump_check(df), iterations),
        "check_user_contiguous_kernel": _time_call(user_kernel, iterations),
        "check_user_timestamp_sorted_kernel": _time_call(jump_kernel, iterations),
        "jump_lengths_indexed": _time_call(
            lambda: _metric_call("jump_lengths", df, presorted=False), iterations
        ),
        "jump_lengths_checked_indexed": _time_call(checked_indexed_jump, iterations),
        "jump_lengths_presorted": _time_call(
            lambda: _metric_call("jump_lengths", df, presorted=True), iterations
        )
        if jump_sorted
        else None,
        "jump_lengths_auto_dispatch": _time_call(auto_jump, iterations),
        "radius_of_gyration_indexed": _time_call(
            lambda: _metric_call("radius_of_gyration", df, presorted=False), iterations
        ),
        "radius_of_gyration_checked_indexed": _time_call(checked_indexed_radius, iterations),
        "radius_of_gyration_presorted": _time_call(
            lambda: _metric_call("radius_of_gyration", df, presorted=True), iterations
        )
        if user_sorted
        else None,
        "radius_of_gyration_auto_dispatch": _time_call(auto_radius, iterations),
    }

    def average(key: str) -> float | None:
        value = metrics.get(key)
        return None if value is None else value["average_seconds"]

    indexed_jump = average("jump_lengths_indexed")
    checked_indexed_jump = average("jump_lengths_checked_indexed")
    indexed_radius = average("radius_of_gyration_indexed")
    checked_indexed_radius = average("radius_of_gyration_checked_indexed")
    check_jump = average("check_user_timestamp_sorted_rust")
    check_user = average("check_user_contiguous_rust")
    kernel_jump = average("check_user_timestamp_sorted_kernel")
    kernel_user = average("check_user_contiguous_kernel")
    presorted_jump = average("jump_lengths_presorted")
    presorted_radius = average("radius_of_gyration_presorted")
    metrics["derived"] = {
        "jump_check_overhead_on_indexed": check_jump,
        "radius_check_overhead_on_indexed": check_user,
        "jump_checked_indexed_overhead_seconds": None
        if checked_indexed_jump is None or indexed_jump is None
        else checked_indexed_jump - indexed_jump,
        "radius_checked_indexed_overhead_seconds": None
        if checked_indexed_radius is None or indexed_radius is None
        else checked_indexed_radius - indexed_radius,
        "jump_presorted_savings_seconds": None
        if presorted_jump is None or indexed_jump is None
        else indexed_jump - presorted_jump,
        "radius_presorted_savings_seconds": None
        if presorted_radius is None or indexed_radius is None
        else indexed_radius - presorted_radius,
        "jump_net_savings_after_check_seconds": None
        if presorted_jump is None or indexed_jump is None or check_jump is None
        else indexed_jump - (check_jump + presorted_jump),
        "radius_net_savings_after_check_seconds": None
        if presorted_radius is None or indexed_radius is None or check_user is None
        else indexed_radius - (check_user + presorted_radius),
        # Same two figures against the kernel-only check cost, i.e. what a
        # measure wrapper that already holds the uid/timestamp buffers would pay.
        "jump_net_savings_after_kernel_check_seconds": None
        if presorted_jump is None or indexed_jump is None or kernel_jump is None
        else indexed_jump - (kernel_jump + presorted_jump),
        "radius_net_savings_after_kernel_check_seconds": None
        if presorted_radius is None or indexed_radius is None or kernel_user is None
        else indexed_radius - (kernel_user + presorted_radius),
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
            "validator": "fastmob._core.validate_grouped_user_timestamps",
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
