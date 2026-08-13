"""Brightkite jump-length benchmark for alternate dataframe backends.

This benchmark deliberately passes each native object directly to fastmob.
It does not convert Modin, Dask, or DuckDB inputs to Pandas: a result marked
``error`` therefore identifies a backend-adapter gap rather than hiding it.

Example::

    .venv/bin/python benchmarks/individual/alternate_backends_jump_lengths.py \
        --sizes 100000 1000000 4000000 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "alternate_backends_jump_lengths.json"
COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]


def load_brightkite_arrow(path: Path) -> Any:
    import pyarrow.csv as csv
    import pyarrow as pa

    return csv.read_csv(
        path,
        read_options=csv.ReadOptions(
            use_threads=True,
            column_names=COLUMNS,
            skip_rows=0,
        ),
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(
            column_types={
                "user": pa.string(),
                "check-in_time": pa.timestamp("us"),
                "latitude": pa.float64(),
                "longitude": pa.float64(),
                "location id": pa.string(),
            },
            timestamp_parsers=["%Y-%m-%dT%H:%M:%SZ"],
        ),
    )


def make_backend(name: str, table: Any, size: int) -> Any:
    sliced = table.slice(0, size)
    if name == "pyarrow":
        return sliced
    if name == "modin":
        import modin.pandas as mpd

        return mpd.DataFrame(sliced.to_pandas())
    if name == "dask":
        import dask.dataframe as dd

        return dd.from_pandas(sliced.to_pandas(), npartitions=max(1, min(16, size // 100_000)))
    if name == "duckdb":
        import duckdb

        return duckdb.from_arrow(sliced)
    raise ValueError(f"unknown backend: {name}")


def benchmark_one(
    backend: str,
    table: Any,
    size: int,
    iterations: int,
) -> dict[str, Any]:
    try:
        from fastmob.measures.individual import jump_lengths

        samples: list[float] = []
        output_type = None
        for _ in range(iterations):
            frame = make_backend(backend, table, size)
            started = time.perf_counter()
            result = jump_lengths(frame, merge=False)
            samples.append(time.perf_counter() - started)
            output_type = f"{type(result).__module__}.{type(result).__name__}"
        return {
            "status": "ok",
            "rows": min(size, table.num_rows),
            "average_seconds": sum(samples) / len(samples),
            "minimum_seconds": min(samples),
            "samples_seconds": samples,
            "output_type": output_type,
        }
    except Exception as exc:  # noqa: BLE001 - backend compatibility is measured
        return {
            "status": "error",
            "rows": min(size, table.num_rows),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 4_000_000])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--modin-engine",
        choices=["python", "dask"],
        default="python",
        help="Modin execution engine; python keeps this direct-input benchmark lightweight.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["pyarrow", "modin", "dask", "duckdb"],
        default=["pyarrow", "modin", "dask", "duckdb"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_path.exists():
        raise SystemExit(f"Dataset not found: {args.data_path}")
    if args.iterations <= 0 or any(size <= 0 for size in args.sizes):
        raise SystemExit("iterations and sizes must be positive")
    os.environ.setdefault("MODIN_ENGINE", args.modin_engine)

    table = load_brightkite_arrow(args.data_path)
    print(f"Brightkite: {table.num_rows:,} rows available")
    results = []
    for size in args.sizes:
        print(f"\nSize {size:,}")
        cases = {}
        for backend in args.backends:
            print(f"  {backend} ...", end=" ", flush=True)
            cases[backend] = benchmark_one(backend, table, size, args.iterations)
            case = cases[backend]
            if case["status"] == "ok":
                print(f"{case['average_seconds']:.6f}s average")
            else:
                print(f"{case['error_type']}: {case['error']}")
        results.append({"size": size, "backends": cases})

    payload = {
        "metadata": {
            "suite": "alternate_backends_jump_lengths",
            "dataset_path": str(args.data_path),
            "iterations": args.iterations,
            "backends": args.backends,
            "modin_engine": args.modin_engine,
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "native_input_direct": True,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
