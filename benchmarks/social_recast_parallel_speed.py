"""Compare RECAST's single-thread and default-Rayon kernel throughput.

The benchmark uses factorized Arrow input directly so it measures event graph
construction, T-RND replicas, and classification rather than dataframe setup.

Usage:
    python benchmarks/social_recast_parallel_speed.py --users 2000 --days 14
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pyarrow as pa


def _input(users: int, days: int, locations: int) -> tuple[pa.Array, pa.Array, pa.Array, pa.Array]:
    day_ms = 86_400_000
    user_ids = list(range(users)) * days
    location_ids = [(user % locations) for _day in range(days) for user in range(users)]
    starts = [day * day_ms + (user % 12) * 60_000 for day in range(days) for user in range(users)]
    ends = [start + 30 * 60_000 for start in starts]
    return (
        pa.array(user_ids, type=pa.uint32()),
        pa.array(location_ids, type=pa.uint32()),
        pa.array(starts, type=pa.int64()),
        pa.array(ends, type=pa.int64()),
    )


def _measure(users: int, days: int, locations: int, replicas: int) -> dict[str, float | int]:
    from fastmob._core import recast_classify

    arrays = _input(users, days, locations)
    started = time.perf_counter()
    result = recast_classify(users, *arrays, 5 * 60_000, 1e-3, replicas, 42)
    elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "edges": len(result[0]), "users": users, "days": days, "replicas": replicas}


def _child(args: argparse.Namespace) -> None:
    print(json.dumps(_measure(args.users, args.days, args.locations, args.replicas)))


def _run_with_threads(args: argparse.Namespace, threads: int | None) -> dict[str, float | int]:
    env = os.environ.copy()
    if threads is None:
        env.pop("RAYON_NUM_THREADS", None)
    else:
        env["RAYON_NUM_THREADS"] = str(threads)
    command = [sys.executable, __file__, "--child", "--users", str(args.users), "--days", str(args.days), "--locations", str(args.locations), "--replicas", str(args.replicas)]
    output = subprocess.check_output(command, text=True, env=env)
    return json.loads(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=2_000)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--locations", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=5)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    single = _run_with_threads(args, 1)
    default = _run_with_threads(args, None)
    print(json.dumps({"single_thread": single, "rayon_default": default, "speedup": single["seconds"] / default["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
