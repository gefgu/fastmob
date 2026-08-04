"""Big-scale RECAST social-relationship classification benchmark against real YJMob100K data.

Mirrors the sibling fastmob_benchmarks repo's daily-motif table benchmark
(`benchmarks/preprocessing_postprocessing/run_table_benchmarks.py`, motif
suite): the same fixed-row-count sizing convention (rows sliced directly off
the YJMob100K parquet with ``.head(n)``, not per-user subsampling), the same
`fastmob` pandas-vs-polars comparison shape, and CSV/LaTeX table output. There
is no comparison-library analogue for RECAST (it is a novel classifier), so
this table only ever has the two fastmob backend columns.

Per row-count case, the untimed input-construction step builds one daily
event-graph input: H3-cell location assignment (``latlng_to_h3``), sorted by
(uid, timestamp), with each row's ``finished_at`` taken from the same user's
next ping (or +30 minutes for a user's last ping in the slice). That yields
the ``Staypoints``/global ``Locations`` pair RECAST requires. Only
``recast_from_staypoints`` itself is timed.

Requires FASTMOB_YJMOB_DATA_PATH; skips cleanly (exit 0) when unset. See
`benchmarks/shared/yjmob.py`.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/social_recast_yjmob_large_scale.py --rows 1000000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_env import detect_cpu_info, get_default_output_dir
from benchmarks.shared.yjmob import skip_reason, yjmob_data_path
from benchmarks.utils import size_label, summarize_times, write_json

DEFAULT_ROWS = [100_000, 1_000_000, 10_000_000, 100_000_000]
H3_RESOLUTION = 9
FULL_ROWS = -1


def _rows_label(rows: int) -> str:
    return "full" if rows == FULL_ROWS else size_label(rows)


def _load_recast_frames(data_path: Path, rows: int, h3_resolution: int) -> tuple[Any, Any]:
    import polars as pl
    from fastmob.preprocessing import latlng_to_h3

    scan = pl.scan_parquet(data_path).select(["uid", "timestamp", "lat", "lon"])
    df = scan.collect() if rows == FULL_ROWS else scan.head(rows).collect()
    df = df.drop_nulls(subset=["uid", "timestamp", "lat", "lon"])
    df = latlng_to_h3(df, resolution=h3_resolution, output_col="location_id")
    df = df.sort(["uid", "timestamp"])
    df = df.with_columns(
        pl.col("timestamp")
        .shift(-1)
        .over("uid")
        .fill_null(pl.col("timestamp") + pl.duration(minutes=30))
        .alias("finished_at")
    ).rename({"timestamp": "started_at"})
    staypoints_df = df.select(["uid", "lat", "lon", "started_at", "finished_at", "location_id"])
    locations_df = df.group_by("location_id").agg(
        pl.col("lat").mean().alias("center_lat"), pl.col("lon").mean().alias("center_lng")
    )
    return staypoints_df, locations_df


def build_recast_input(data_path: Path, rows: int, backend: str, h3_resolution: int) -> dict[str, Any]:
    from fastmob.core import Locations, Staypoints

    staypoints_df, locations_df = _load_recast_frames(data_path, rows, h3_resolution)
    n_rows = len(staypoints_df)
    n_locations = len(locations_df)
    if backend == "pandas":
        staypoints_df = staypoints_df.to_pandas()
        locations_df = locations_df.to_pandas()

    staypoints = Staypoints(staypoints_df)
    locations = Locations(locations_df, scope="global")
    return {"staypoints": staypoints, "locations": locations, "n_rows": n_rows, "n_locations": n_locations}


def benchmark_recast(
    data_path: Path,
    rows: int,
    backend: str,
    iterations: int,
    *,
    h3_resolution: int = H3_RESOLUTION,
    min_minutes_for_encounter: int = 5,
    p_rnd: float = 1e-3,
    random_replicates: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    from fastmob.social import recast_from_staypoints

    prepared = build_recast_input(data_path, rows, backend, h3_resolution)
    staypoints, locations = prepared["staypoints"], prepared["locations"]

    times: list[float] = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = recast_from_staypoints(
            staypoints,
            locations,
            min_minutes_for_encounter=min_minutes_for_encounter,
            p_rnd=p_rnd,
            random_replicates=random_replicates,
            seed=seed,
        )
        times.append(time.perf_counter() - start)

    n_rows = prepared["n_rows"]
    stats = summarize_times(times)
    rows_per_second = n_rows / stats["minimum_seconds"] if stats["minimum_seconds"] else None
    return {
        "benchmark": "recast_from_staypoints",
        "backend": backend,
        "rows": rows,
        "n_rows": n_rows,
        "size_label": _rows_label(rows),
        "n_locations": prepared["n_locations"],
        "time_steps": result.time_steps if result is not None else 0,
        "edge_count": len(result.classes) if result is not None else 0,
        **stats,
        "rows_per_second": rows_per_second,
    }


def tex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    path: Path, *, caption: str, label: str, columns: list[tuple[str, str]], rows: list[dict[str, Any]], note: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\scriptsize",
        r"\centering",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{header}}}" for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row[key]) for key, _ in columns) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{\linewidth}",
            r"\vspace{1ex}",
            rf"\footnotesize{{\textit{{Note:}} {tex_escape(note)}}}",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_seconds(row: dict[str, Any] | None) -> str:
    if row is None or row.get("minimum_seconds") is None:
        return "n/a"
    return f"{row['minimum_seconds']:.4f}"


def write_recast_tables(results: list[dict[str, Any]], tables_dir: Path) -> None:
    by_key = {(r["backend"], r["rows"]): r for r in results}
    sizes = sorted({r["rows"] for r in results}, key=lambda value: (value == FULL_ROWS, value))
    table_rows = []
    for rows in sizes:
        pandas_row = by_key.get(("pandas", rows))
        polars_row = by_key.get(("polars", rows))
        table_rows.append(
            {
                "size": _rows_label(rows),
                "fastmob_pandas_seconds": fmt_seconds(pandas_row),
                "fastmob_polars_seconds": fmt_seconds(polars_row),
            }
        )
    write_csv_rows(tables_dir / "social_recast_creation_table.csv", table_rows)
    write_latex_table(
        tables_dir / "social_recast_creation_table.tex",
        caption="Runtime benchmark for RECAST social-relationship classification on YJMob-backed inputs.",
        label="tab:social_recast_creation_benchmark",
        columns=[
            ("size", "Size"),
            ("fastmob_pandas_seconds", "Fastmob pandas s"),
            ("fastmob_polars_seconds", "Fastmob polars s"),
        ],
        rows=table_rows,
        note=(
            "Timings are minimum-of-iterations, profiler-free, operation-only perf_counter "
            "measurements of recast_from_staypoints after input construction (H3 location "
            "assignment and Staypoints/global-Locations framing); there is no comparison-"
            "library analogue for RECAST."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)
    parser.add_argument("--min-minutes-for-encounter", type=int, default=5)
    parser.add_argument("--p-rnd", type=float, default=1e-3)
    parser.add_argument("--random-replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tables-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = args.data_path or yjmob_data_path()
    if data_path is None:
        print(skip_reason())
        return 0

    backends = ["pandas", "polars"] if args.backend == "both" else [args.backend]
    results = []
    for rows in sorted(args.rows, key=lambda value: (value == FULL_ROWS, value)):
        for backend in backends:
            print(f"Benchmarking recast_from_staypoints: rows={_rows_label(rows)}, backend={backend}...")
            results.append(
                benchmark_recast(
                    data_path,
                    rows,
                    backend,
                    args.iterations,
                    h3_resolution=args.h3_resolution,
                    min_minutes_for_encounter=args.min_minutes_for_encounter,
                    p_rnd=args.p_rnd,
                    random_replicates=args.random_replicates,
                    seed=args.seed,
                )
            )

    output_dir = get_default_output_dir()
    payload = {
        "metadata": {
            "benchmark": "social.recast_from_staypoints",
            "dataset": "yjmob100k",
            "h3_resolution": args.h3_resolution,
            "min_minutes_for_encounter": args.min_minutes_for_encounter,
            "p_rnd": args.p_rnd,
            "random_replicates": args.random_replicates,
            "seed": args.seed,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
        },
        "results": results,
    }

    output_path = args.output or (output_dir / "fastmob_social_recast_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")

    tables_dir = args.tables_dir or (output_dir / "tables")
    write_recast_tables(results, tables_dir)
    print(f"Wrote tables under {tables_dir}")

    for r in results:
        extra = r.get("rows_per_second")
        extra_str = f"rows/sec={extra:.0f}" if extra else ""
        print(
            f"  size={r['size_label']:>6} backend={r['backend']:<7} rows={r['n_rows']:>10} "
            f"edges={r['edge_count']:>8} min_seconds={r['minimum_seconds']:.4f} {extra_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
