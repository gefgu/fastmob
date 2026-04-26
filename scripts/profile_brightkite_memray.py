#!/usr/bin/env python
"""Run pytest-memray flamegraph profiling for Brightkite-backed skmob2 workloads."""

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


DEFAULT_OUTPUT_DIR = Path(".profiles") / "memray"
PROFILE_TEST = "tests/profiling/test_brightkite_memray.py::test_memray_brightkite_workload"


@dataclass(frozen=True)
class ProfileCommand:
    workload: str
    bin_dir: Path
    bin_prefix: str
    bin_path: Path
    flamegraph_path: Path
    pytest_command: list[str]
    flamegraph_command: list[str]


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
    pytest_bin: str = "pytest",
    memray_bin: str = "memray",
) -> ProfileCommand:
    bin_dir = output_dir / "bins"
    flamegraph_dir = output_dir / "flamegraphs"
    bin_prefix = workload
    bin_path = bin_dir / f"{bin_prefix}.bin"
    flamegraph_path = flamegraph_dir / f"{workload}.html"
    pytest_command = [
        pytest_bin,
        PROFILE_TEST,
        "--profile-workload",
        workload,
        "--profile-rows",
        str(rows),
        "--profile-backend",
        backend,
        "--memray",
        "--native",
        "--memray-bin-path",
        str(bin_dir),
        "--memray-bin-prefix",
        bin_prefix,
    ]
    flamegraph_command = [
        memray_bin,
        "flamegraph",
        "-o",
        str(flamegraph_path),
        str(bin_path),
    ]
    return ProfileCommand(
        workload=workload,
        bin_dir=bin_dir,
        bin_prefix=bin_prefix,
        bin_path=bin_path,
        flamegraph_path=flamegraph_path,
        pytest_command=pytest_command,
        flamegraph_command=flamegraph_command,
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
    pytest_bin = getattr(args, "pytest_bin", "pytest")
    memray_bin = getattr(args, "memray_bin", "memray")

    if not args.dry_run:
        _require_executable(pytest_bin, "pytest")
        _require_executable(memray_bin, "memray")

    manifest_rows: list[dict[str, Any]] = []
    exit_code = 0
    for workload in selected:
        profile = build_profile_command(
            workload,
            rows=args.rows,
            backend=args.backend,
            output_dir=output_dir,
            pytest_bin=pytest_bin,
            memray_bin=memray_bin,
        )
        profile.bin_dir.mkdir(parents=True, exist_ok=True)
        profile.flamegraph_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        status = "dry-run"
        returncode = 0
        error = ""
        actual_bin_path = profile.bin_path
        if args.dry_run:
            print(" ".join(profile.pytest_command))
            print(" ".join(profile.flamegraph_command))
        else:
            _clear_previous_bins(profile)
            print(f"Profiling {workload} -> {profile.flamegraph_path}")
            completed = subprocess.run(profile.pytest_command, check=False)
            returncode = completed.returncode
            status = "ok" if returncode == 0 else "failed"
            if returncode == 0:
                try:
                    actual_bin_path = _find_memray_bin(profile)
                    flamegraph_command = [
                        *profile.flamegraph_command[:-1],
                        str(actual_bin_path),
                    ]
                    completed = subprocess.run(flamegraph_command, check=False)
                    returncode = completed.returncode
                    status = "ok" if returncode == 0 else "failed"
                    if returncode != 0:
                        error = f"memray flamegraph exited with {returncode}"
                except RuntimeError as exc:
                    status = "failed"
                    returncode = 1
                    error = str(exc)
            else:
                error = f"pytest-memray exited with {returncode}"

            if returncode != 0:
                exit_code = returncode
                if not args.continue_on_error:
                    elapsed = time.perf_counter() - started
                    manifest_rows.append(
                        _manifest_row(args, profile, status, returncode, elapsed, error, actual_bin_path)
                    )
                    break

        elapsed = time.perf_counter() - started
        manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error, actual_bin_path))

    write_manifest(output_dir, manifest_rows)
    return exit_code


def _require_executable(executable: str, package_name: str) -> None:
    if shutil.which(executable) is None:
        raise SystemExit(
            f"{executable!r} is not installed or not on PATH. Install the dev extras first or install {package_name!r}."
        )


def _clear_previous_bins(profile: ProfileCommand) -> None:
    for path in profile.bin_dir.glob(f"{profile.bin_prefix}*.bin"):
        path.unlink()


def _find_memray_bin(profile: ProfileCommand) -> Path:
    if profile.bin_path.exists():
        return profile.bin_path
    matches = sorted(profile.bin_dir.glob(f"{profile.bin_prefix}*.bin"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"pytest-memray did not create a bin file for {profile.workload!r}")
    names = ", ".join(str(path) for path in matches)
    raise RuntimeError(f"pytest-memray created multiple bin files for {profile.workload!r}: {names}")


def _manifest_row(
    args: argparse.Namespace,
    profile: ProfileCommand,
    status: str,
    returncode: int,
    elapsed: float,
    error: str,
    bin_path: Path,
) -> dict[str, Any]:
    flamegraph_command = [*profile.flamegraph_command[:-1], str(bin_path)]
    return {
        "workload": profile.workload,
        "rows": args.rows,
        "backend": args.backend,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(elapsed, 6),
        "bin_path": str(bin_path),
        "flamegraph_path": str(profile.flamegraph_path),
        "pytest_command": " ".join(profile.pytest_command),
        "flamegraph_command": " ".join(flamegraph_command),
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Run `maturin develop` before profiling. The runner always passes "
            "`--native` to pytest-memray so Rust frames from skmob2._core can appear "
            "when native symbols are available."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--workload", action="append", help="Workload to profile; repeatable.")
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pytest-bin", default="pytest")
    parser.add_argument("--memray-bin", default="memray")
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
        help="Print pytest-memray commands and write manifests without launching pytest.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, workload in workload_registry().items():
            print(f"{name}\t{workload.dataset}\t{workload.description}")
        return 0
    return run_profiles(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
