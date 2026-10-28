"""Eager vs streaming radius of gyration: wall time and peak RSS.

Writes a Brightkite prefix to Parquet once per size, then measures, each in a
fresh subprocess so peak RSS is not polluted by earlier runs:

- ``eager``: ``pl.read_parquet`` + ``fastmob.radius_of_gyration``
- ``lazy``:  ``pl.scan_parquet`` + ``fastmob.radius_of_gyration_lazy``

Run from the repository root:

    python benchmarks/individual/radius_of_gyration_lazy.py --sizes 1000000,4000000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
DEFAULT_CACHE_DIR = REPO_ROOT / "benchmarks" / "cache" / "radius_of_gyration_lazy"

_CHILD = """
import json, sys, time
import polars as pl
import fastmob

mode, path, chunk_size = sys.argv[1], sys.argv[2], sys.argv[3]
chunk_size = None if chunk_size == "auto" else int(chunk_size)
started = time.perf_counter()
if mode == "eager":
    result = fastmob.radius_of_gyration(pl.read_parquet(path))
else:
    result = fastmob.radius_of_gyration_lazy(pl.scan_parquet(path), chunk_size=chunk_size)
elapsed = time.perf_counter() - started
# VmHWM, not ru_maxrss: Linux carries ru_maxrss over from the parent across
# fork/exec, while VmHWM belongs to the fresh address space exec creates.
with open("/proc/self/status") as status:
    peak_kib = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
print(json.dumps({"seconds": elapsed, "peak_rss_mib": peak_kib / 1024, "users": result.height}))
"""


def prepare_parquet(data_path: Path, rows: int, cache_dir: Path) -> Path:
    """Write the first `rows` Brightkite check-ins (uid/datetime/lat/lng) to Parquet, once."""
    import polars as pl

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"brightkite_{rows}.parquet"
    if not path.exists():
        pl.read_csv(
            data_path,
            separator="\t",
            has_header=False,
            new_columns=["uid", "datetime", "lat", "lng"],
            columns=[0, 1, 2, 3],
            n_rows=rows,
        ).write_parquet(path)
    return path


def measure(mode: str, path: Path, chunk_size: int | None) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, str(path), "auto" if chunk_size is None else str(chunk_size)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sizes", type=_parse_sizes, default=[1_000_000, 4_000_000])
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for rows in args.sizes:
        path = prepare_parquet(args.data_path, rows, args.cache_dir)
        for mode in ("eager", "lazy"):
            runs = [measure(mode, path, args.chunk_size) for _ in range(args.repeats)]
            best = min(runs, key=lambda run: run["seconds"])
            entry = {"rows": rows, "mode": mode, **best, "peak_rss_mib": max(r["peak_rss_mib"] for r in runs)}
            results.append(entry)
            print(
                f"{rows:>10,} {mode:<5} {entry['seconds']:8.3f}s  "
                f"peak {entry['peak_rss_mib']:8.1f} MiB  users {entry['users']:,}"
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
