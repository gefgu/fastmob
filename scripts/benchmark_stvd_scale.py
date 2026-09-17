"""Synthetic YJMob100K-shaped scale benchmark for ``mean_area_volume``.

Generates staypoint-like data (many users x many days, multi-bin stay
durations) at a controllable row count and times
``fastmob.measures.collective.stvd.mean_area_volume`` end to end, reporting
wall-clock time and peak RSS. Not a pytest-benchmark suite: this is a
one-shot scale probe (rows can go into the tens/hundreds of millions), which
doesn't fit pytest-benchmark's repeated-trial model.

Usage:
    .venv/bin/python scripts/benchmark_stvd_scale.py --rows 4000000
    .venv/bin/python scripts/benchmark_stvd_scale.py --rows 100000000 --users 100000 --days 75
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import pandas as pd


def generate_visits(
    rows: int,
    *,
    users: int,
    areas: int,
    days: int,
    avg_duration_min: float,
    seed: int,
) -> pd.DataFrame:
    """Build a synthetic visits table shaped like a real staypoint export.

    Each row is one stay: a user in an area for a lognormal-ish duration
    (mean ``avg_duration_min``), scattered across ``days`` calendar days.
    Multi-bin durations exercise the 10-minute presence expansion the same
    way real dwell-time staypoints would, unlike degenerate start==end rows.
    """
    rng = np.random.default_rng(seed)

    user_id = rng.integers(0, users, size=rows, dtype=np.int64)
    area = rng.integers(0, areas, size=rows, dtype=np.int64)
    day_offset = rng.integers(0, days, size=rows, dtype=np.int64)
    minute_of_day = rng.integers(0, 24 * 60, size=rows, dtype=np.int64)
    start_seconds = day_offset * 86_400 + minute_of_day * 60

    # Lognormal duration centered near avg_duration_min, clipped to a sane range.
    duration_min = rng.lognormal(mean=np.log(max(avg_duration_min, 1.0)), sigma=0.8, size=rows)
    duration_min = np.clip(duration_min, 1.0, 8 * 60.0)
    duration_seconds = (duration_min * 60).astype(np.int64)

    base = np.datetime64("2024-01-01T00:00:00", "s")
    start_ts = base + start_seconds.astype("timedelta64[s]")
    end_ts = start_ts + duration_seconds.astype("timedelta64[s]")

    return pd.DataFrame(
        {
            "area": area,
            "user_id": user_id,
            "start_timestamp": pd.to_datetime(start_ts),
            "end_timestamp": pd.to_datetime(end_ts),
        }
    )


def peak_rss_gb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS; this project only targets Linux dev boxes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4_000_000)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--areas", type=int, default=20_000)
    parser.add_argument("--days", type=int, default=75)
    parser.add_argument("--avg-duration-min", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gen_start = time.perf_counter()
    visits = generate_visits(
        args.rows,
        users=args.users,
        areas=args.areas,
        days=args.days,
        avg_duration_min=args.avg_duration_min,
        seed=args.seed,
    )
    gen_elapsed = time.perf_counter() - gen_start
    gen_rss = peak_rss_gb()

    from fastmob.measures.collective.stvd import mean_area_volume

    call_start = time.perf_counter()
    result = mean_area_volume(visits)
    call_elapsed = time.perf_counter() - call_start
    call_rss = peak_rss_gb()

    print(f"rows={args.rows} users={args.users} areas={args.areas} days={args.days}")
    print(f"generation: {gen_elapsed:.2f}s, peak_rss={gen_rss:.2f}GB")
    print(f"mean_area_volume: {call_elapsed:.2f}s, peak_rss={call_rss:.2f}GB")
    print(f"output rows: {len(result)}")
    print(f"TOTAL: {gen_elapsed + call_elapsed:.2f}s, peak_rss={call_rss:.2f}GB")


if __name__ == "__main__":
    main()
