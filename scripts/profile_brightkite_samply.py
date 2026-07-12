#!/usr/bin/env python
"""Run samply profiling for Brightkite-backed workloads."""

from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.profile_brightkite_common import (
    implementations,
    profile_jobs,
    require_executable,
    select_workloads,
    write_manifest,
)
from tests.profiling.brightkite_workloads import (
    DEFAULT_ROWS,
    DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
    JUMP_LENGTHS_ENTRYPOINTS,
    workload_registry,
)


DEFAULT_OUTPUT_DIR = Path(".profiles") / "samply"
SCOPES = ("function", "full")


@dataclass(frozen=True)
class ProfileCommand:
    workload: str
    implementation: str
    scope: str
    output_path: Path
    command: list[str]
    child_command: list[str] | None = None


def _workload_command(
    workload: str,
    *,
    rows: int,
    backend: str,
    implementation: str,
    prepared_child: bool,
    jump_lengths_entrypoint: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tests.profiling.brightkite_workloads",
        "--workload",
        workload,
        "--rows",
        str(rows),
        "--backend",
        backend,
        "--implementation",
        implementation,
        "--jump-lengths-entrypoint",
        jump_lengths_entrypoint,
    ]
    if prepared_child:
        command.append("--prepared-child")
    return command


def build_profile_command(
    workload: str,
    *,
    rows: int,
    backend: str,
    output_dir: Path,
    rate: int,
    samply_bin: str = "samply",
    implementation: str = "fastmob",
    scope: str = "function",
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> ProfileCommand:
    profile_dir = output_dir / implementation
    suffix = "" if scope == "function" else ".full"
    output_path = profile_dir / f"{workload}{suffix}.json.gz"
    child_command = _workload_command(
        workload,
        rows=rows,
        backend=backend,
        implementation=implementation,
        prepared_child=scope == "function",
        jump_lengths_entrypoint=jump_lengths_entrypoint,
    )

    if scope == "function":
        command = [
            samply_bin,
            "record",
            "--save-only",
            "--unstable-presymbolicate",
            "--no-open",
            "--rate",
            str(rate),
            "-o",
            str(output_path),
            "--pid",
            "<prepared-child-pid>",
        ]
    else:
        command = [
            samply_bin,
            "record",
            "--save-only",
            "--unstable-presymbolicate",
            "--no-open",
            "--rate",
            str(rate),
            "-o",
            str(output_path),
            "--",
            *child_command,
        ]

    return ProfileCommand(
        workload=workload,
        implementation=implementation,
        scope=scope,
        output_path=output_path,
        command=command,
        child_command=child_command if scope == "function" else None,
    )


def run_profiles(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samply_bin = getattr(args, "samply_bin", "samply")
    scope = getattr(args, "scope", "function")

    if not args.dry_run:
        require_executable(samply_bin, "samply", install_hint="Install with: cargo install --locked samply")

    manifest_rows: list[dict[str, Any]] = []
    exit_code = 0
    jobs = profile_jobs(args.implementation, args.workload, select_workloads)
    for implementation, workload in jobs:
        profile = build_profile_command(
            workload,
            rows=args.rows,
            backend=args.backend,
            output_dir=output_dir,
            rate=args.rate,
            samply_bin=samply_bin,
            implementation=implementation,
            scope=scope,
            jump_lengths_entrypoint=getattr(args, "jump_lengths_entrypoint", DEFAULT_JUMP_LENGTHS_ENTRYPOINT),
        )
        profile.output_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        status = "dry-run"
        returncode = 0
        error = ""
        if args.dry_run:
            if profile.child_command:
                print(" ".join(profile.child_command))
            print(" ".join(profile.command))
        else:
            print(f"Profiling {implementation}:{workload} ({scope}) -> {profile.output_path}")
            if scope == "function":
                returncode, error = _run_function_profile(profile, profile.command)
            else:
                completed = subprocess.run(profile.command, check=False)
                returncode = completed.returncode
                error = "" if returncode == 0 else f"samply exited with {returncode}"

            status = "ok" if returncode == 0 else "failed"
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


def _run_function_profile(profile: ProfileCommand, command_template: list[str]) -> tuple[int, str]:
    if profile.child_command is None:
        raise RuntimeError("function profiling requires a prepared child command")

    child = subprocess.Popen(
        profile.child_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    try:
        assert child.stdout is not None
        ready, error = _wait_until_ready(child)
        if not ready:
            return 1, error

        command = [str(child.pid) if part == "<prepared-child-pid>" else part for part in command_template]
        profiler = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
        attached, attach_error = _wait_for_profiler_attach(profiler)
        if not attached:
            returncode = profiler.wait()
            return returncode or 1, attach_error

        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        child_returncode = child.wait()
        remaining_stderr = profiler.communicate()[1]
        if remaining_stderr:
            sys.stderr.write(remaining_stderr)
        profiler_returncode = profiler.returncode
        if child_returncode != 0:
            return child_returncode, f"prepared child exited with {child_returncode}"
        if profiler_returncode != 0:
            return profiler_returncode, f"samply exited with {profiler_returncode}"
        return 0, ""
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


_SAMPLY_ATTACH_MARKERS = ("Recording", "Sampling", "attached", "Press Ctrl")


def _wait_for_profiler_attach(profiler: subprocess.Popen[str], timeout: float = 10.0) -> tuple[bool, str]:
    if profiler.stderr is None:
        time.sleep(1.0)
        return profiler.poll() is None, "samply exited before the workload was released"

    deadline = time.monotonic() + timeout
    stderr_fd = profiler.stderr.fileno()
    while time.monotonic() < deadline:
        if profiler.poll() is not None:
            rest = profiler.stderr.read()
            if rest:
                sys.stderr.write(rest)
            return False, f"samply exited before attaching (returncode={profiler.returncode})"

        ready, _, _ = select.select([stderr_fd], [], [], 0.1)
        if not ready:
            continue

        line = profiler.stderr.readline()
        if not line:
            continue
        sys.stderr.write(line)
        if any(marker in line for marker in _SAMPLY_ATTACH_MARKERS):
            return True, ""

    return True, ""


def _wait_until_ready(child: subprocess.Popen[str]) -> tuple[bool, str]:
    assert child.stdout is not None
    while True:
        line = child.stdout.readline()
        if not line:
            returncode = child.wait()
            return False, f"prepared child exited before ready (returncode={returncode})"
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "ready":
            return True, ""


def _manifest_row(
    args: argparse.Namespace,
    profile: ProfileCommand,
    status: str,
    returncode: int,
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    row = {
        "workload": profile.workload,
        "rows": args.rows,
        "backend": args.backend,
        "implementation": profile.implementation,
        "scope": profile.scope,
        "profiled_phase": "function" if profile.scope == "function" else "full",
        "jump_lengths_entrypoint": getattr(args, "jump_lengths_entrypoint", DEFAULT_JUMP_LENGTHS_ENTRYPOINT),
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(elapsed, 6),
        "output_path": str(profile.output_path),
        "command": " ".join(profile.command),
        "error": error,
    }
    if profile.child_command:
        row["child_command"] = " ".join(profile.child_command)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Function scope is the default: data loading, imports, and setup happen "
            "before samply attaches to the prepared child. Use --scope full for "
            "whole-process profiling that includes load and import time."
        ),
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--workload", action="append", help="Workload to profile; repeatable.")
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rate", type=int, default=1000)
    parser.add_argument("--samply-bin", default="samply")
    parser.add_argument("--backend", choices=["pandas", "polars"], default="pandas")
    parser.add_argument("--scope", choices=SCOPES, default="function")
    parser.add_argument("--implementation", choices=("fastmob", "skmob", "both"), default="fastmob")
    parser.add_argument(
        "--jump-lengths-entrypoint",
        choices=JUMP_LENGTHS_ENTRYPOINTS,
        default=DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
        help="fastmob jump_lengths TrajDataFrame entrypoint to profile.",
    )
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
        help="Print samply commands and write manifests without launching samply.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for implementation in implementations(args.implementation):
            for name, workload in workload_registry(implementation).items():
                print(f"{name}\t{workload.dataset}\t{implementation}\t{workload.description}")
        return 0
    return run_profiles(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
