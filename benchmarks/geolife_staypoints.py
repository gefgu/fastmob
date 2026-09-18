"""Benchmark Fastmob and Trackintel staypoint detection on full GeoLife.

Install the optional benchmark dependencies first:
``pip install geopandas trackintel``. The timed region excludes data loading
and GeoDataFrame construction, and both implementations use a 100 m radius,
20-minute dwell, and 15-minute observation gap.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from benchmarks.benchmark_env import detect_cpu_info
from benchmarks.utils import positive_int, write_json


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def time_call(call: Callable[[], Any], iterations: int) -> tuple[dict[str, Any], Any]:
    call()  # warm-up
    times: list[float] = []
    result: Any = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return {
        "iterations_completed": iterations,
        "times_seconds": times,
        "minimum_seconds": min(times),
        "median_seconds": _median(times),
        "average_seconds": sum(times) / len(times),
    }, result


def compare_counts(fastmob_stays: Any, trackintel_stays: Any) -> dict[str, Any]:
    """Report semantic comparability without asserting identical detectors."""
    fastmob_frame = pd.DataFrame(fastmob_stays)
    trackintel_frame = pd.DataFrame(trackintel_stays)
    return {
        "fastmob_staypoints": len(fastmob_frame),
        "trackintel_staypoints": len(trackintel_frame),
        "relative_stay_count_difference": abs(len(fastmob_frame) - len(trackintel_frame)) / max(1, len(trackintel_frame)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import fastmob
    import geopandas as gpd
    import trackintel as ti

    trajectory = fastmob.io.load_geolife_trajectories(args.data_dir).to_pandas()
    trajectory["datetime"] = pd.to_datetime(trajectory["datetime"], errors="coerce", utc=True)
    trajectory = trajectory.dropna(subset=["uid", "datetime", "lat", "lng"]).reset_index(drop=True)
    trackintel_input = trajectory.rename(columns={"uid": "user_id", "datetime": "tracked_at"})
    positionfixes = ti.Positionfixes(
        gpd.GeoDataFrame(trackintel_input, geometry=gpd.points_from_xy(trackintel_input["lng"], trackintel_input["lat"]), crs="EPSG:4326")
    )

    def fastmob_call() -> Any:
        return fastmob.stay_locations(trajectory, spatial_radius_km=0.1, minutes_for_a_stop=20, no_data_for_minutes=15, leaving_time=True)

    def trackintel_call() -> Any:
        return positionfixes.generate_staypoints(method="sliding", dist_threshold=100, time_threshold=20, gap_threshold=15, include_last=True, n_jobs=-1)[1]

    fastmob_timing, fastmob_stays = time_call(fastmob_call, args.iterations)
    trackintel_timing, trackintel_stays = time_call(trackintel_call, args.iterations)
    return {
        "metadata": {
            "dataset": "Microsoft GeoLife GPS Trajectories 1.3",
            "rows": len(trajectory),
            "cpu": detect_cpu_info(),
            "memory_total_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2),
            "platform": platform.platform(),
            "python_version": sys.version,
            "packages": {"fastmob": _version("fastmob"), "trackintel": _version("trackintel")},
            "detector": {"radius_m": 100, "minutes_for_a_stop": 20, "gap_minutes": 15},
        },
        "results": {
            "fastmob": fastmob_timing,
            "trackintel": trackintel_timing,
            "median_speedup_vs_trackintel": trackintel_timing["median_seconds"] / fastmob_timing["median_seconds"],
            "output_comparability": compare_counts(fastmob_stays, trackintel_stays),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/geolife_staypoints.json"))
    parser.add_argument("--iterations", type=positive_int, default=7)
    args = parser.parse_args()
    write_json(run(args), args.output)


if __name__ == "__main__":
    main()
