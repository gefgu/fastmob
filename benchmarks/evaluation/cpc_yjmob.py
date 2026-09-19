"""Profile YJMob CPC through materialized OD flows and direct Trips inputs.

Run from the repository root with:
    .venv/bin/python benchmarks/evaluation/cpc_yjmob.py

The two comparison inputs are disjoint user cohorts with approximately the
requested number of raw pings each. Full per-user traces are retained.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import narwhals as nw
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastmob import FlowDataFrame
from fastmob._core import common_part_of_commuters as rust_cpc
from fastmob.measures.collective.od import od_matrix
from fastmob.measures.evaluation import common_part_of_commuters
from fastmob.preprocessing import trajectory_to_od, trajectory_to_trips
from fastmob.utils._common import _as_arrow, _joint_factorize_arrow_values

from benchmarks.benchmark_env import detect_cpu_info

DEFAULT_DATA_PATH = REPO_ROOT.parent / "fastmob_benchmarks" / "data" / "raw" / "yjmob_wgs84_simple.parquet"
DEFAULT_SIZES = (1_000_000, 10_000_000)
RESOLUTIONS = (7, 8, 9)
DATA_COLUMNS = ("uid", "timestamp", "lat", "lon")


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


class PeakRssSampler:
    """Sample current RSS during an operation and report growth over baseline."""

    def __init__(self, interval_seconds: float = 0.01):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.baseline = 0
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop_event.wait(self.interval_seconds):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __exit__(self, *_exc):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def additional_peak_mb(self) -> float:
        return max(0, self.peak - self.baseline) / (1024 * 1024)


def choose_user_cohorts(data_path: Path, target_rows: int):
    import polars as pl

    # YJMob is UID-major. Maintaining first appearance order lets the two
    # cohorts be adjacent, non-overlapping sets while keeping complete traces.
    counts = (
        pl.scan_parquet(data_path)
        .select("uid")
        .group_by("uid", maintain_order=True)
        .len(name="rows")
        .collect(engine="streaming")
    )
    ids = counts.get_column("uid").to_list()
    sizes = counts.get_column("rows").to_list()
    cohorts: list[list[Any]] = []
    actual_rows: list[int] = []
    offset = 0
    for _ in range(2):
        selected: list[Any] = []
        total = 0
        while offset < len(ids) and total < target_rows:
            selected.append(ids[offset])
            total += sizes[offset]
            offset += 1
        if total < target_rows:
            raise ValueError(f"YJMob does not contain two disjoint {target_rows:,}-row cohorts")
        cohorts.append(selected)
        actual_rows.append(total)
    return cohorts, actual_rows


def load_cohort(data_path: Path, user_ids: list[Any]):
    import polars as pl

    return (
        pl.scan_parquet(data_path)
        .filter(pl.col("uid").is_in(user_ids))
        .select(DATA_COLUMNS)
        .collect(engine="streaming")
    )


def cpc_component_times(first: Any, second: Any, *, trips: bool, iterations: int) -> dict[str, Any]:
    def inputs(value: Any):
        df = nw.from_native(value.df, eager_only=True)
        if trips:
            origins = df.get_column("origin_location_id")
            destinations = df.get_column("destination_location_id")
            weights_a = weights_b = None
        else:
            origins = df.get_column("origin")
            destinations = df.get_column("destination")
            weights_a = pc.cast(_as_arrow(df.get_column("flow")), pa.float64())
            weights_b = weights_a
        return origins, destinations, weights_a, weights_b

    a_origin, a_destination, weights_a, _ = inputs(first)
    b_origin, b_destination, weights_b, _ = inputs(second)
    factor_times: list[float] = []
    kernel_times: list[float] = []
    scores: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        factorized = _joint_factorize_arrow_values(a_origin, a_destination, b_origin, b_destination)
        factor_times.append(time.perf_counter() - start)
        oa, da, ob, db = factorized
        start = time.perf_counter()
        scores.append(float(rust_cpc(oa, da, weights_a, ob, db, weights_b)))
        kernel_times.append(time.perf_counter() - start)
    return {
        "endpoint_factorization": summarize(factor_times),
        "rust_cpc_kernel": summarize(kernel_times),
        "score": scores[-1],
    }


def timed_pair_build(builder: Callable, left: Any, right: Any, resolution: int, iterations: int):
    left_times: list[float] = []
    right_times: list[float] = []
    last_left = last_right = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_left = builder(left, resolution=resolution)
        left_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        last_right = builder(right, resolution=resolution)
        right_times.append(time.perf_counter() - start)
    return last_left, last_right, summarize(left_times), summarize(right_times)


def run_direct(left: Any, right: Any, resolution: int, iterations: int):
    with PeakRssSampler() as memory:
        a, b, a_build, b_build = timed_pair_build(trajectory_to_trips, left, right, resolution, iterations)
        cpc_times: list[float] = []
        scores: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            scores.append(common_part_of_commuters(a, b))
            cpc_times.append(time.perf_counter() - start)
        details = cpc_component_times(a, b, trips=True, iterations=iterations)
    result = {
        "trajectory_to_trips_left": a_build,
        "trajectory_to_trips_right": b_build,
        "cpc_wrapper": summarize(cpc_times),
        "cpc_components": details,
        "total_median_seconds": a_build["median_seconds"] + b_build["median_seconds"] + statistics.median(cpc_times),
        "additional_peak_rss_mb": memory.additional_peak_mb,
        "score": scores[-1],
        "left_trip_rows": len(a.df),
        "right_trip_rows": len(b.df),
    }
    return result, (a, b)


def run_materialized(left: Any, right: Any, resolution: int, iterations: int):
    od_left_times: list[float] = []
    od_right_times: list[float] = []
    wrap_left_times: list[float] = []
    wrap_right_times: list[float] = []
    cpc_times: list[float] = []
    last_a = last_b = None
    with PeakRssSampler() as memory:
        for _ in range(iterations):
            start = time.perf_counter()
            od_a = trajectory_to_od(left, resolution=resolution)
            od_left_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            od_b = trajectory_to_od(right, resolution=resolution)
            od_right_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            flow_a = FlowDataFrame(od_a, origin="origin", destination="destination", flow="count")
            wrap_left_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            flow_b = FlowDataFrame(od_b, origin="origin", destination="destination", flow="count")
            wrap_right_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            common_part_of_commuters(flow_a, flow_b)
            cpc_times.append(time.perf_counter() - start)
            last_a, last_b = flow_a, flow_b
        score = common_part_of_commuters(last_a, last_b)
        details = cpc_component_times(last_a, last_b, trips=False, iterations=iterations)
    medians = [
        statistics.median(od_left_times),
        statistics.median(od_right_times),
        statistics.median(wrap_left_times),
        statistics.median(wrap_right_times),
        statistics.median(cpc_times),
    ]
    return {
        "trajectory_to_od_left": summarize(od_left_times),
        "trajectory_to_od_right": summarize(od_right_times),
        "flow_dataframe_left": summarize(wrap_left_times),
        "flow_dataframe_right": summarize(wrap_right_times),
        "cpc_wrapper": summarize(cpc_times),
        "cpc_components": details,
        "total_median_seconds": sum(medians),
        "additional_peak_rss_mb": memory.additional_peak_mb,
        "score": score,
        "left_od_edges": len(last_a.df),
        "right_od_edges": len(last_b.df),
    }


def standalone_od_aggregation(trips_a: Any, trips_b: Any, iterations: int):
    times: list[float] = []
    with PeakRssSampler() as memory:
        for _ in range(iterations):
            start = time.perf_counter()
            od_matrix(trips_a.df, origin_col="origin_location_id", destination_col="destination_location_id")
            od_matrix(trips_b.df, origin_col="origin_location_id", destination_col="destination_location_id")
            times.append(time.perf_counter() - start)
    return {**summarize(times), "additional_peak_rss_mb": memory.additional_peak_mb}


def format_report(payload: dict[str, Any]) -> str:
    rows = [
        "# YJMob CPC benchmark report",
        "",
        f"Run: {payload['metadata']['timestamp_utc']}  ",
        f"CPU: {payload['metadata']['cpu_info']['model']} ({payload['metadata']['cpu_info']['cores']} logical CPUs)  ",
        f"Python: {payload['metadata']['python']}  ",
        f"Data: `{payload['metadata']['data_path']}`",
        "",
        "The 1M/10M labels are approximate raw-ping counts per side. Each comparison uses two disjoint cohorts with complete user traces. Peak RSS is sampled additional process RSS above the already loaded cohort frames; it is approximate.",
        "",
        "| Tier | Input load/slice (s) | H3 | Direct Trips total (s) | OD DataFrame total (s) | OD / Direct time | Direct peak RSS delta (MB) | OD peak RSS delta (MB) | Trips / side | OD edges / side | CPC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        direct = item["direct"]
        materialized = item["materialized"]
        speedup = materialized["total_median_seconds"] / direct["total_median_seconds"]
        rows.append(
            f"| {item['size_label']} | {item['input_loading_seconds']:.3f} | {item['resolution']} | "
            f"{direct['total_median_seconds']:.3f} | "
            f"{materialized['total_median_seconds']:.3f} | {speedup:.2f}× | "
            f"{direct['additional_peak_rss_mb']:.1f} | {materialized['additional_peak_rss_mb']:.1f} | "
            f"{direct['left_trip_rows']:,} / {direct['right_trip_rows']:,} | "
            f"{materialized['left_od_edges']:,} / {materialized['right_od_edges']:,} | {direct['score']:.6f} |"
        )
    rows.extend(["", "## Timing breakdown", ""])
    for item in payload["results"]:
        rows.extend(
            [
                f"### {item['size_label']} pings per side, H3 {item['resolution']}",
                "",
                "| Path / phase | Median seconds |",
                "|---|---:|",
            ]
        )
        for name, result in item["direct"].items():
            if isinstance(result, dict) and "median_seconds" in result:
                rows.append(f"| Direct: {name} | {result['median_seconds']:.4f} |")
        for name, result in item["materialized"].items():
            if isinstance(result, dict) and "median_seconds" in result:
                rows.append(f"| OD path: {name} | {result['median_seconds']:.4f} |")
        for path_name in ("direct", "materialized"):
            components = item[path_name]["cpc_components"]
            rows.append(
                f"| {path_name}: endpoint factorization | {components['endpoint_factorization']['median_seconds']:.4f} |"
            )
            rows.append(f"| {path_name}: Rust CPC kernel | {components['rust_cpc_kernel']['median_seconds']:.4f} |")
        rows.append("")
    rows.extend(
        [
            "## Interpretation",
            "",
            "Direct total includes building paired Trips for both cohorts and the public CPC call. OD total includes `trajectory_to_od` for both cohorts, FlowDataFrame construction, and the public CPC call. Endpoint factorization and Rust kernel rows are diagnostic decompositions and are not added again to those totals.",
            "",
            "In this run, the direct Trips path is slower and uses more additional RSS in all six tier/resolution cases. Avoiding `od_matrix` does not offset carrying and factorizing every paired trip through CPC; the materialized path's CPC sees only unique OD edges. The direct helper remains useful when callers need individual consecutive-ping pairs, but this benchmark does not show a performance or memory win for CPC as currently implemented.",
            "",
        ]
    )
    return "\n".join(rows)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "benchmarks/results/cpc_yjmob.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "benchmarks/CPC_YJMOB_BENCHMARK_REPORT.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_path.is_file():
        raise SystemExit(f"YJMob parquet not found: {args.data_path}")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    import polars as pl

    results = []
    for target_rows in args.sizes:
        started = time.perf_counter()
        cohorts, requested_row_counts = choose_user_cohorts(args.data_path, target_rows)
        inputs = [load_cohort(args.data_path, user_ids) for user_ids in cohorts]
        input_loading_seconds = time.perf_counter() - started
        actual_row_counts = [len(df) for df in inputs]
        if actual_row_counts != requested_row_counts:
            raise RuntimeError("cohort row count changed between sizing and loading")
        print(
            f"Loaded {target_rows:,} tier: {actual_row_counts[0]:,} and {actual_row_counts[1]:,} rows "
            f"in {input_loading_seconds:.2f}s",
            flush=True,
        )
        for resolution in args.resolutions:
            direct, direct_trips = run_direct(inputs[0], inputs[1], resolution, args.iterations)
            standalone_od = standalone_od_aggregation(direct_trips[0], direct_trips[1], args.iterations)
            del direct_trips
            # Allow direct-path output buffers to be reclaimed before measuring
            # the materialized OD path's incremental RSS.
            gc.collect()
            materialized = run_materialized(inputs[0], inputs[1], resolution, args.iterations)
            materialized["standalone_od_matrix_aggregation_both_sides"] = standalone_od
            if not np.isclose(direct["score"], materialized["score"], rtol=1e-10, atol=1e-12):
                raise AssertionError(
                    f"CPC scores differ at {target_rows:,}/{resolution}: "
                    f"direct={direct['score']} materialized={materialized['score']}"
                )
            results.append(
                {
                    "size": target_rows,
                    "size_label": f"{target_rows // 1_000_000}M",
                    "resolution": resolution,
                    "input_rows": actual_row_counts,
                    "n_users": [len(cohort) for cohort in cohorts],
                    "input_loading_seconds": input_loading_seconds,
                    "direct": direct,
                    "materialized": materialized,
                }
            )
            del direct, materialized, standalone_od
            gc.collect()
            print(f"Completed {target_rows // 1_000_000}M, H3 {resolution}", flush=True)
        del inputs, cohorts
        gc.collect()

    payload = {
        "metadata": {
            "benchmark": "YJMob CPC direct Trips vs materialized OD flows",
            "dataset": "YJMob100K",
            "data_path": str(args.data_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_info": detect_cpu_info(),
            "python": sys.version.split()[0],
            "polars": pl.__version__,
            "iterations": args.iterations,
            "resolutions": args.resolutions,
            "input_selection": "two adjacent, disjoint UID cohorts; complete user trajectories",
            "memory_metric": "maximum sampled RSS above source cohorts already resident in process",
        },
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(format_report(payload), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
