#!/usr/bin/env python
"""Run Scalene profiling for Brightkite-backed workloads."""

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

from tests.profiling.brightkite_workloads import (
    DEFAULT_ROWS,
    IMPLEMENTATIONS,
    _skmob_workload_candidates,
    workload_registry,
)


DEFAULT_OUTPUT_DIR = Path(".profiles") / "scalene"
DEFAULT_WORKLOADS = ["jump_lengths"]
SCALENE_VIEW_OUTPUT = "scalene-profile.html"


@dataclass(frozen=True)
class ProfileCommand:
    workload: str
    implementation: str
    json_path: Path
    html_path: Path
    profile_dir: Path
    run_command: list[str]
    html_command: list[str]


def _implementations(requested: str) -> list[str]:
    if requested == "both":
        return list(IMPLEMENTATIONS)
    return [requested]


def select_workloads(
    requested: list[str] | None,
    implementation: str = "skmob2",
    *,
    allow_unavailable: bool = False,
) -> list[str]:
    registry = workload_registry(implementation)
    if allow_unavailable and implementation == "skmob":
        registry = _skmob_workload_candidates()
    if implementation == "skmob" and not registry:
        raise SystemExit(
            "No skmob workloads are available because scikit-mobility is not importable. "
            "Install the comparison extra with `uv pip install -e '.[dev-skmob]'` "
            "or run `bash tests/setup_env.sh --skmob`, then rerun this profile."
        )
    workloads = requested or DEFAULT_WORKLOADS
    unknown = sorted(set(workloads) - set(registry))
    if unknown:
        available = ", ".join(registry) or "none"
        raise SystemExit(
            f"Unknown workload(s) for {implementation}: {', '.join(unknown)}. Available: {available}"
        )
    return list(workloads)


def build_profile_command(
    workload: str,
    *,
    rows: int,
    backend: str,
    output_dir: Path,
    scalene_bin: str = "scalene",
    implementation: str = "skmob2",
) -> ProfileCommand:
    profile_dir = output_dir / implementation
    json_path = profile_dir / f"{workload}.json"
    html_path = profile_dir / f"{workload}.html"
    workload_command = [
        "tests/profiling/brightkite_workloads.py",
        "---",
        "--workload",
        workload,
        "--rows",
        str(rows),
        "--backend",
        backend,
        "--implementation",
        implementation,
        "--scalene-function-profile",
    ]
    run_command = [
        scalene_bin,
        "run",
        "--off",
        "-o",
        str(json_path),
        *workload_command,
    ]
    html_command = [
        scalene_bin,
        "view",
        "--standalone",
        json_path.name,
    ]
    return ProfileCommand(
        workload=workload,
        implementation=implementation,
        json_path=json_path,
        html_path=html_path,
        profile_dir=profile_dir,
        run_command=run_command,
        html_command=html_command,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalene_bin = getattr(args, "scalene_bin", "scalene")

    if not args.dry_run:
        _require_executable(scalene_bin, "scalene")

    manifest_rows: list[dict[str, Any]] = []
    exit_code = 0
    jobs = [
        (implementation, workload)
        for implementation in _implementations(args.implementation)
        for workload in select_workloads(args.workload, implementation, allow_unavailable=args.dry_run)
    ]
    for implementation, workload in jobs:
        profile = build_profile_command(
            workload,
            rows=args.rows,
            backend=args.backend,
            output_dir=output_dir,
            scalene_bin=scalene_bin,
            implementation=implementation,
        )
        profile.profile_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        status = "dry-run"
        returncode = 0
        error = ""
        if args.dry_run:
            print(" ".join(profile.run_command))
            print(" ".join(profile.html_command))
        else:
            _clear_previous_outputs(profile)
            print(f"Profiling {implementation}:{workload} -> {profile.json_path}")
            completed = subprocess.run(profile.run_command, check=False)
            returncode = completed.returncode
            status = "ok" if returncode == 0 else "failed"
            if returncode == 0:
                print(f"Rendering Scalene HTML {implementation}:{workload} -> {profile.html_path}")
                returncode, error = _render_html(profile)
                status = "ok" if returncode == 0 else "failed"
            else:
                error = f"scalene run exited with {returncode}"

            if returncode != 0:
                exit_code = returncode
                if not args.continue_on_error:
                    elapsed = time.perf_counter() - started
                    manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error))
                    break

        elapsed = time.perf_counter() - started
        manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error))

    write_manifest(output_dir, manifest_rows)
    return exit_code


def _require_executable(executable: str, package_name: str) -> None:
    if shutil.which(executable) is None:
        raise SystemExit(
            f"{executable!r} is not installed or not on PATH. Install the dev extras first or install {package_name!r}."
        )


def _clear_previous_outputs(profile: ProfileCommand) -> None:
    for path in (profile.json_path, profile.html_path, profile.profile_dir / SCALENE_VIEW_OUTPUT):
        if path.exists():
            path.unlink()


def _render_html(profile: ProfileCommand) -> tuple[int, str]:
    completed = subprocess.run(profile.html_command, check=False, cwd=profile.profile_dir)
    if completed.returncode != 0:
        return completed.returncode, f"scalene view exited with {completed.returncode}"

    generated_path = profile.profile_dir / SCALENE_VIEW_OUTPUT
    if not generated_path.exists():
        return 1, f"scalene view did not create {generated_path}"
    generated_path.replace(profile.html_path)
    return 0, ""


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
        "implementation": profile.implementation,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(elapsed, 6),
        "json_path": str(profile.json_path),
        "html_path": str(profile.html_path),
        "run_command": " ".join(profile.run_command),
        "html_command": " ".join(profile.html_command),
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "By default, Scalene profiles the jump_lengths workload for both skmob2 and skmob. "
            "The workload setup is prepared before Scalene profiling is enabled."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--workload", action="append", help="Workload to profile; repeatable.")
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scalene-bin", default="scalene")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="pandas")
    parser.add_argument("--implementation", choices=("skmob2", "skmob", "both"), default="both")
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
        help="Print Scalene commands and write manifests without launching Scalene.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for implementation in _implementations(args.implementation):
            for name, workload in workload_registry(implementation).items():
                print(f"{name}\t{workload.dataset}\t{implementation}\t{workload.description}")
        return 0
    return run_profiles(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
