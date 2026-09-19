"""Full parquet-to-RECAST benchmark; each size runs in a fresh process.

Use --extension to compare isolated release binaries without installing them.
Results include exact output digests, stage timings, and process peak RSS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    if args.extension:
        spec = importlib.util.spec_from_file_location("fastmob._core", args.extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["fastmob._core"] = module
    import polars as pl
    from fastmob.core import Locations, Staypoints
    from fastmob.preprocessing import latlng_to_h3
    from fastmob.social import recast, recast_from_staypoints

    stages = {}
    started = last = time.perf_counter()

    def mark(name):
        nonlocal last
        now = time.perf_counter()
        stages[name] = now - last
        last = now

    df = pl.scan_parquet(args.data_path).select("uid", "timestamp", "lat", "lon").head(args.rows[0]).collect()
    df = df.drop_nulls()
    mark("load")
    df = latlng_to_h3(df, resolution=9, output_col="location_id")
    mark("h3")
    df = df.sort(["uid", "timestamp"]).with_columns(
        pl.col("timestamp").shift(-1).over("uid")
        .fill_null(pl.col("timestamp") + pl.duration(minutes=30)).alias("finished_at")
    ).rename({"timestamp": "started_at"})
    locations = df.group_by("location_id").agg(
        pl.col("lat").mean().alias("center_lat"), pl.col("lon").mean().alias("center_lng")
    )
    rows, users = len(df), df["uid"].n_unique()
    staypoints = Staypoints(df.select("uid", "lat", "lon", "started_at", "finished_at", "location_id"))
    locations = Locations(locations, scope="global")
    del df
    mark("intervals_and_locations")
    original_prepare = recast._prepare

    def timed_prepare(*values):
        t = time.perf_counter()
        result = original_prepare(*values)
        stages["prepare"] = time.perf_counter() - t
        return result

    recast._prepare = timed_prepare
    result = recast_from_staypoints(staypoints, locations, random_replicates=args.replicas)
    mark("classify_including_prepare")
    elapsed = time.perf_counter() - started
    digests = {}
    for name in ("source_users", "target_users", "edge_persistence", "topological_overlap", "classes"):
        array = getattr(result, name)
        digests[name] = hashlib.sha256(array.to_numpy(zero_copy_only=False).tobytes()).hexdigest()
    return {
        "rows": rows,
        "users": users,
        "days": result.time_steps,
        "edges": len(result.classes),
        "seconds": elapsed,
        "stages": stages,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "persistence_threshold": result.persistence_threshold,
        "overlap_threshold": result.overlap_threshold,
        "digests": digests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=os.environ.get("FASTMOB_YJMOB_DATA_PATH"))
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[1_000_000, 10_000_000, 30_000_000, 100_000_000])
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.data_path or not args.data_path.is_file():
        parser.error("provide --data-path or FASTMOB_YJMOB_DATA_PATH")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    results = []
    for rows in args.rows:
        for iteration in range(args.iterations):
            command = [sys.executable, __file__, "--worker", "--data-path", str(args.data_path),
                       "--rows", str(rows), "--replicas", str(args.replicas)]
            if args.extension:
                command += ["--extension", str(args.extension.resolve())]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=args.timeout,
                    env={**os.environ, "FASTMOB_RECAST_PROFILE": "1"},
                )
                result = json.loads(completed.stdout) if completed.returncode == 0 else {"error": completed.stderr}
                result["profile"] = completed.stderr.splitlines()
            except subprocess.TimeoutExpired as error:
                result = {"timeout_seconds": args.timeout, "profile": (error.stderr or b"").decode().splitlines()}
            result.update(requested_rows=rows, iteration=iteration)
            results.append(result)
            print(json.dumps(result), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "results": results,
                            "extension": str(args.extension),
                            "data_path": str(args.data_path),
                            "replicas": args.replicas,
                            "cpu_count": os.cpu_count(),
                            "threads": {
                                key: os.environ.get(key) for key in ("RAYON_NUM_THREADS", "POLARS_MAX_THREADS")
                            },
                        },
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
