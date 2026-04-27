#!/usr/bin/env python
"""Run py-spy profiling for Brightkite-backed skmob2 workloads."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.profiling.brightkite_workloads import DEFAULT_ROWS, workload_registry


DEFAULT_OUTPUT_DIR = Path(".profiles") / "py-spy"


@dataclass(frozen=True)
class ProfileCommand:
    workload: str
    output_path: Path
    speedscope_output_path: Path
    command: list[str]
    speedscope_command: list[str]


def select_workloads(requested: list[str] | None) -> list[str]:
    registry = workload_registry()
    if not requested:
        return list(registry)
    unknown = sorted(set(requested) - set(registry))
    if unknown:
        available = ", ".join(registry)
        raise SystemExit(f"Unknown workload(s): {', '.join(unknown)}. Available: {available}")
    return requested


def build_profile_command(
    workload: str,
    *,
    rows: int,
    backend: str,
    output_dir: Path,
    rate: int,
    py_spy_bin: str = "py-spy",
) -> ProfileCommand:
    output_path = output_dir / f"{workload}.svg"
    speedscope_output_path = output_dir / f"{workload}.speedscope.json"
    workload_command = [
        sys.executable,
        "-m",
        "tests.profiling.brightkite_workloads",
        "--workload",
        workload,
        "--rows",
        str(rows),
        "--backend",
        backend,
    ]
    command = [
        py_spy_bin,
        "record",
        "--native",
        "--format",
        "flamegraph",
        "--rate",
        str(rate),
        "-o",
        str(output_path),
        "--",
        *workload_command,
    ]
    speedscope_command = [
        py_spy_bin,
        "record",
        "--native",
        "--format",
        "speedscope",
        "--rate",
        str(rate),
        "-o",
        str(speedscope_output_path),
        "--",
        *workload_command,
    ]
    return ProfileCommand(
        workload=workload,
        output_path=output_path,
        speedscope_output_path=speedscope_output_path,
        command=command,
        speedscope_command=speedscope_command,
    )


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "manifest.json"
    csv_path = output_dir / "manifest.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        csv_path.write_text("")
        return
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_profiles(args: argparse.Namespace) -> int:
    selected = select_workloads(args.workload)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    py_spy_bin = getattr(args, "py_spy_bin", "py-spy")

    if shutil.which(py_spy_bin) is None and not args.dry_run:
        raise SystemExit(f"{py_spy_bin!r} is not installed or not on PATH. Install the dev extras first.")

    manifest_rows: list[dict[str, Any]] = []
    exit_code = 0
    for workload in selected:
        profile = build_profile_command(
            workload,
            rows=args.rows,
            backend=args.backend,
            output_dir=output_dir,
            rate=args.rate,
            py_spy_bin=py_spy_bin,
        )
        started = time.perf_counter()
        status = "dry-run"
        returncode = 0
        error = ""
        if args.dry_run:
            print(" ".join(profile.command))
            print(" ".join(profile.speedscope_command))
        else:
            print(f"Profiling {workload} -> {profile.output_path}")
            completed = subprocess.run(profile.command, check=False)
            returncode = completed.returncode
            status = "ok" if returncode == 0 else "failed"
            if returncode != 0:
                error = f"py-spy exited with {returncode}"
                exit_code = returncode
                if not args.continue_on_error:
                    elapsed = time.perf_counter() - started
                    manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error))
                    break
            else:
                print(f"Profiling {workload} -> {profile.speedscope_output_path}")
                completed = subprocess.run(profile.speedscope_command, check=False)
                returncode = completed.returncode
                status = "ok" if returncode == 0 else "failed"
                if returncode != 0:
                    error = f"py-spy speedscope exited with {returncode}"
                    exit_code = returncode
                    if not args.continue_on_error:
                        elapsed = time.perf_counter() - started
                        manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error))
                        break

        elapsed = time.perf_counter() - started
        manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error))

    write_manifest(output_dir, manifest_rows)
    return exit_code


def _manifest_row(
    args: argparse.Namespace,
    profile: ProfileCommand,
    status: str,
    returncode: int,
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    return {
        "workload": profile.workload,
        "rows": args.rows,
        "backend": args.backend,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(elapsed, 6),
        "output_path": str(profile.output_path),
        "speedscope_output_path": str(profile.speedscope_output_path),
        "command": " ".join(profile.command),
        "speedscope_command": " ".join(profile.speedscope_command),
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Run `maturin develop` before profiling. The runner writes SVG "
            "flamegraphs and Speedscope JSON files, and always passes "
            "`--native` to py-spy so Rust frames from skmob2._core can appear "
            "when native symbols are available."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--workload", action="append", help="Workload to profile; repeatable.")
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--py-spy-bin", default="py-spy")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="pandas")
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Continue profiling remaining workloads after a failure.",
    )
    parser.add_argument(
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Stop after the first failed workload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print py-spy commands and write manifests without launching py-spy.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, workload in workload_registry().items():
            print(f"{name}\t{workload.dataset}\t{workload.description}")
        return 0
    return run_profiles(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
