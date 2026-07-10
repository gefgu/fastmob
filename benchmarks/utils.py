"""Shared helpers for standalone benchmark suites."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable


def size_label(size: int) -> str:
    if size >= 1_000_000 and size % 1_000_000 == 0:
        return f"{size // 1_000_000}M"
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def summarize_times(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"average_seconds": None, "minimum_seconds": None}
    return {"average_seconds": sum(times) / len(times), "minimum_seconds": min(times)}


def summarize_memory(peak_memory_mb: list[float]) -> dict[str, float | None]:
    if not peak_memory_mb:
        return {
            "average_peak_memory_mb": None,
            "minimum_peak_memory_mb": None,
            "maximum_peak_memory_mb": None,
        }
    return {
        "average_peak_memory_mb": sum(peak_memory_mb) / len(peak_memory_mb),
        "minimum_peak_memory_mb": min(peak_memory_mb),
        "maximum_peak_memory_mb": max(peak_memory_mb),
    }


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def merge_payload(existing: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial benchmark payload into an existing result file."""
    merged = dict(existing)
    merged["metadata"] = {**existing.get("metadata", {}), **partial.get("metadata", {})}

    results_by_label = {result["label"]: dict(result) for result in existing.get("results", [])}
    for partial_result in partial.get("results", []):
        label = partial_result["label"]
        if label not in results_by_label:
            results_by_label[label] = dict(partial_result)
            continue

        merged_result = results_by_label[label]
        merged_result.update({key: value for key, value in partial_result.items() if key != "metrics"})
        merged_metrics = dict(merged_result.get("metrics", {}))
        merged_metrics.update(partial_result.get("metrics", {}))
        merged_result["metrics"] = merged_metrics

    merged["results"] = sorted(results_by_label.values(), key=lambda result: result.get("size", 0))
    return merged


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def call_benchmark_func(func: Callable[..., Any], input_value: Any, kwargs: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        return func(input_value, **kwargs)


def run_timed_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print("    Warming up...")
    call_benchmark_func(func, make_input(), kwargs)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        start = time.perf_counter()
        call_benchmark_func(func, make_input(), kwargs)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        print(f"    Round {i + 1}: {duration:.4f} seconds")

    summary = summarize_times(times)
    print(f"    Average Time: {summary['average_seconds']:.4f} s")
    print(f"    Minimum Time: {summary['minimum_seconds']:.4f} s")
    return {
        "status": "ok",
        "times_seconds": times,
        "iterations_completed": len(times),
        **summary,
    }


def run_memory_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    import contextlib
    import tempfile

    import memray

    peak_memory_mb: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp_path = f.name
        os.unlink(tmp_path)  # memray requires the destination to not exist yet
        try:
            with memray.Tracker(tmp_path, native_traces=False):
                call_benchmark_func(func, make_input(), kwargs)
            reader = memray.FileReader(tmp_path)
            peak_bytes = sum(
                r.size
                for r in reader.get_high_watermark_allocation_records(merge_threads=True)
            )
            peak_mb = peak_bytes / (1024 * 1024)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
        peak_memory_mb.append(peak_mb)
        print(f"    Round {i + 1}: {peak_mb:.4f} MB peak (memray)")

    summary = summarize_memory(peak_memory_mb)
    print(f"    Average Peak Memory: {summary['average_peak_memory_mb']:.4f} MB")
    print(f"    Minimum Peak Memory: {summary['minimum_peak_memory_mb']:.4f} MB")
    print(f"    Maximum Peak Memory: {summary['maximum_peak_memory_mb']:.4f} MB")
    return {
        "status": "ok",
        "peak_memory_mb": peak_memory_mb,
        "current_memory_mb": [],
        "iterations_completed": len(peak_memory_mb),
        **summary,
    }


def run_profiled_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    profile: str,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    if profile == "memory":
        return run_memory_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)
    return run_timed_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)


def empty_profile_result(status: str, reason: str, profile: str) -> dict[str, Any]:
    if profile == "memory":
        return {
            "status": status,
            "reason": reason,
            "peak_memory_mb": [],
            "current_memory_mb": [],
            "iterations_completed": 0,
            **summarize_memory([]),
        }
    return {
        "status": status,
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def skipped_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("skipped", reason, profile)


def error_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("error", reason, profile)


def _profiled_call_worker(
    result_queue: Any,
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    profile: str,
    iterations: int,
    sleep_seconds: float,
) -> None:
    try:
        result_queue.put(
            (
                "ok",
                run_profiled_call(
                    func,
                    make_input,
                    kwargs,
                    profile=profile,
                    iterations=iterations,
                    sleep_seconds=sleep_seconds,
                ),
            )
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_profiled_call_isolated(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    profile: str,
    iterations: int,
    sleep_seconds: float,
    retries: int,
    case_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    ctx = mp.get_context("fork")
    attempts = retries + 1
    last_reason = "benchmark did not produce a result"
    for attempt in range(1, attempts + 1):
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_profiled_call_worker,
            args=(result_queue, func, make_input, kwargs, profile, iterations, sleep_seconds),
        )
        process.start()
        process.join(case_timeout_seconds)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            if timed_out:
                status, payload = "error", f"case exceeded timeout of {case_timeout_seconds:g} seconds"
            else:
                status, payload = "error", f"child process exited with code {process.exitcode}"

        result_queue.close()
        result_queue.join_thread()

        if process.exitcode == 0 and status == "ok":
            return payload

        if status == "ok":
            last_reason = f"child process exited with code {process.exitcode} after reporting success"
        else:
            last_reason = str(payload)
            if process.exitcode not in (0, None):
                last_reason = f"{last_reason}; child process exited with code {process.exitcode}"

        if attempt < attempts:
            print(f"    attempt {attempt} failed: {last_reason}")
            print(f"    retrying ({attempt + 1}/{attempts})...")

    print(f"    error after {attempts} attempt(s): {last_reason}")
    return error_result(last_reason, profile)


def concrete_backends(library: str, backend: str) -> Iterable[str | None]:
    if library != "fkmob":
        return (None,)
    if backend == "both":
        return ("pandas", "polars")
    return (backend,)


def concrete_input_orders(input_order: str) -> Iterable[str]:
    if input_order == "both":
        return ("raw", "sorted")
    return (input_order,)
