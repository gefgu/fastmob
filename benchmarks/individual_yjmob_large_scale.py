"""Big-scale profile-classification benchmark against the real YJMob100K dataset.

`compute_profiles` (item 3 of the citybehavex migration plan) composes
several per-user measures plus a scikit-learn KMeans clustering step -- this
is the heaviest single-function benchmark in the migration since it's not one
kernel but a full pipeline. YJMob100K ships raw pings, not stay-level visits,
so each user's consecutive pings are treated as a stay (H3 cell as location,
own timestamp as start, next ping's timestamp as end).

Requires FASTMOB_YJMOB_DATA_PATH; skips cleanly (exit 0) when unset. See
`benchmarks/shared/yjmob.py`.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/individual_yjmob_large_scale.py --n-users 1000 5000 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402
from benchmarks.shared.yjmob import load_yjmob, skip_reason, yjmob_data_path  # noqa: E402
from benchmarks.utils import size_label, summarize_times, write_json  # noqa: E402

# compute_profiles's cost is dominated by intermittance_and_degree_of_return's
# impute_gaps=True path (a pure-Python per-row 5-minute trajectory expansion,
# a pre-existing characteristic of that function, not something introduced by
# this port) -- observed ~30s for 1,000 users / 1.2M rows on this machine, so
# the full 100k-user YJMob100K scale is impractical here; these defaults stay
# at a realistic-report scale rather than the dataset's full size.
DEFAULT_N_USERS = [1_000, 5_000]
H3_RESOLUTION = 9


def _build_visits(data_path: Path, n_users: int):
    import polars as pl

    from fastmob.preprocessing import latlng_to_h3

    df = load_yjmob(data_path, n_users=n_users)
    df = latlng_to_h3(df, resolution=H3_RESOLUTION, output_col="location_id")
    df = df.sort(["uid", "timestamp"])
    df = df.with_columns(
        pl.col("timestamp").shift(-1).over("uid").fill_null(pl.col("timestamp") + pl.duration(minutes=30)).alias(
            "end_timestamp"
        )
    )
    return df.rename({"timestamp": "start_timestamp"}).select(
        ["uid", "start_timestamp", "end_timestamp", "location_id"]
    )


def benchmark_compute_profiles(data_path: Path, n_users: int, iterations: int) -> dict:
    from fastmob.measures.individual import compute_profiles

    visits = _build_visits(data_path, n_users)
    n_rows = len(visits)

    times: list[float] = []
    n_profiled = 0
    for _ in range(iterations):
        start = time.perf_counter()
        result = compute_profiles(visits, n_clusters=3, random_state=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        n_profiled = len(result)

    stats = summarize_times(times)
    return {
        "n_users": n_users,
        "n_rows": n_rows,
        "size_label": size_label(n_rows),
        **stats,
        "n_users_profiled": n_profiled,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-users", type=int, nargs="+", default=DEFAULT_N_USERS)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = args.data_path or yjmob_data_path()
    if data_path is None:
        print(skip_reason())
        return 0

    results = []
    for n_users in sorted(args.n_users):
        print(f"Benchmarking compute_profiles: n_users={n_users}...")
        results.append(benchmark_compute_profiles(data_path, n_users, args.iterations))

    payload = {
        "metadata": {
            "benchmark": "individual.compute_profiles",
            "dataset": "yjmob100k",
            "h3_resolution": H3_RESOLUTION,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_compute_profiles_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        print(
            f"  n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
            f"min_seconds={r['minimum_seconds']:.4f} profiled={r['n_users_profiled']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
