#!/usr/bin/env python
"""Run Scalene profiling for Brightkite-backed workloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
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
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_REDUCED_CPU_SAMPLING_RATE = 0.001


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
    profile_scope: str | None = None,
    reduced: bool = False,
    cpu_only: bool = False,
    cpu_sampling_rate: float | None = None,
) -> ProfileCommand:
    profile_dir = output_dir / implementation
    if implementation == "skmob2" and backend != "pandas":
        profile_dir = profile_dir / backend
    json_path = profile_dir / (f".{workload}.scalene.tmp.json" if reduced else f"{workload}.json")
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
        "--profile-only",
        profile_scope or implementation,
    ]
    if cpu_only:
        run_command.append("--cpu-only")
    if cpu_sampling_rate is not None:
        run_command.extend(["--cpu-sampling-rate", f"{cpu_sampling_rate:g}"])
    run_command.extend(["--off", "-o", str(json_path), *workload_command])
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
            profile_scope=getattr(args, "profile_scope", None),
            reduced=getattr(args, "reduced", False),
            cpu_only=_cpu_only(args),
            cpu_sampling_rate=_cpu_sampling_rate(args),
        )
        profile.profile_dir.mkdir(parents=True, exist_ok=True)

        reduced_json_path = ""

        started = time.perf_counter()
        status = "dry-run"
        returncode = 0
        error = ""
        timed_out = False
        if args.dry_run:
            print(" ".join(profile.run_command))
            if getattr(args, "reduced", False):
                reduced_path = profile.profile_dir / f"{workload}.reduced.json"
                print(" ".join(_reducer_command(args, profile, reduced_path)))
            else:
                print(" ".join(profile.html_command))
        else:
            _clear_previous_outputs(profile)
            print(f"Profiling {implementation}:{workload} -> {profile.json_path}")
            returncode, timed_out, error = _run_command(
                profile.run_command,
                timeout_seconds=getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            )
            status = "timed_out" if timed_out else ("ok" if returncode == 0 else "failed")
            if returncode == 0:
                if getattr(args, "reduced", False):
                    returncode, error, reduced_json_path = _reduce_profile(args, profile)
                    status = "ok" if returncode == 0 else "failed"
                    _remove_path(profile.json_path)
                else:
                    print(f"Rendering Scalene HTML {implementation}:{workload} -> {profile.html_path}")
                    returncode, timed_out, error = _render_html(
                        profile,
                        timeout_seconds=getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
                    )
                    status = "timed_out" if timed_out else ("ok" if returncode == 0 else "failed")
            elif not timed_out:
                error = f"scalene run exited with {returncode}"
            elif getattr(args, "reduced", False):
                _remove_path(profile.json_path)

            if returncode != 0:
                exit_code = returncode
                if not args.continue_on_error:
                    elapsed = time.perf_counter() - started
                    manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error, reduced_json_path))
                    break

        elapsed = time.perf_counter() - started
        manifest_rows.append(_manifest_row(args, profile, status, returncode, elapsed, error, reduced_json_path))

    write_manifest(output_dir, manifest_rows)
    return exit_code


def _require_executable(executable: str, package_name: str) -> None:
    if shutil.which(executable) is None:
        raise SystemExit(
            f"{executable!r} is not installed or not on PATH. Install the dev extras first or install {package_name!r}."
        )


def _run_command(
    command: list[str],
    *,
    timeout_seconds: float | None,
    cwd: Path | None = None,
) -> tuple[int, bool, str]:
    timeout = None if timeout_seconds is None or timeout_seconds <= 0 else timeout_seconds
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return process.wait(timeout=timeout), False, ""
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        _terminate_command_output_processes(command)
        timeout_label = f"{timeout_seconds:g}" if timeout_seconds is not None else "unknown"
        return 124, True, f"scalene run timed out after {timeout_label}s"


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    pids = [*reversed(_descendant_pids(process.pid)), process.pid]

    for pid in pids:
        _signal_pid(pid, signal.SIGTERM)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        for pid in pids:
            _signal_pid(pid, signal.SIGKILL)
        process.wait()


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _descendant_pids(parent_pid: int) -> list[int]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    children_by_parent: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = (int(parts[0]), int(parts[1]))
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    stack = list(children_by_parent.get(parent_pid, ()))
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children_by_parent.get(pid, ()))
    return descendants


def _terminate_command_output_processes(command: list[str]) -> None:
    output_path = _command_output_path(command)
    if output_path is None:
        return

    pids = _pids_with_command_fragment(str(output_path))
    for pid in pids:
        _signal_pid(pid, signal.SIGTERM)
    time.sleep(0.2)
    for pid in _pids_with_command_fragment(str(output_path)):
        _signal_pid(pid, signal.SIGKILL)


def _command_output_path(command: list[str]) -> Path | None:
    try:
        output_index = command.index("-o") + 1
    except ValueError:
        return None
    if output_index >= len(command):
        return None
    return Path(command[output_index])


def _pids_with_command_fragment(fragment: str) -> list[int]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    current_pid = os.getpid()
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if fragment in command:
            pids.append(pid)
    return pids


def _clear_previous_outputs(profile: ProfileCommand) -> None:
    for path in (
        profile.json_path,
        profile.profile_dir / f"{profile.workload}.json",
        profile.profile_dir / f".{profile.workload}.scalene.json.json",
        profile.html_path,
        profile.profile_dir / f"{profile.workload}.reduced.json",
        profile.profile_dir / SCALENE_VIEW_OUTPUT,
    ):
        if path.exists():
            path.unlink()


def _reduce_profile(args: argparse.Namespace, profile: ProfileCommand) -> tuple[int, str, str]:
    reduced_path = profile.profile_dir / f"{profile.workload}.reduced.json"
    reduced_json_cmd = _reducer_command(args, profile, reduced_path)
    print(f"Generating reduced JSON {profile.implementation}:{profile.workload} -> {reduced_path}")
    red = subprocess.run(reduced_json_cmd, check=False)
    if red.returncode != 0:
        return red.returncode, f"reducer exited with {red.returncode}", ""
    return 0, "", str(reduced_path)


def _reducer_command(args: argparse.Namespace, profile: ProfileCommand, reduced_path: Path) -> list[str]:
    reducer = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "skills"
        / "profile-python-scalene"
        / "scripts"
        / "reduce_scalene_json.py"
    )
    return [
        sys.executable,
        str(reducer),
        str(profile.json_path),
        "--top",
        str(getattr(args, "reduced_top", 20)),
        "-o",
        str(reduced_path),
    ]


def _remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _render_html(profile: ProfileCommand, *, timeout_seconds: float | None) -> tuple[int, bool, str]:
    returncode, timed_out, error = _run_command(
        profile.html_command,
        timeout_seconds=timeout_seconds,
        cwd=profile.profile_dir,
    )
    if timed_out:
        return returncode, timed_out, error.replace("scalene run", "scalene view")
    if returncode != 0:
        return returncode, False, f"scalene view exited with {returncode}"

    generated_path = profile.profile_dir / SCALENE_VIEW_OUTPUT
    if not generated_path.exists():
        return 1, False, f"scalene view did not create {generated_path}"
    generated_path.replace(profile.html_path)
    return 0, False, ""


def _manifest_row(
    args: argparse.Namespace,
    profile: ProfileCommand,
    status: str,
    returncode: int,
    elapsed: float,
    error: str,
    reduced_json_path: str = "",
) -> dict[str, Any]:
    reduced_only = getattr(args, "reduced", False)
    return {
        "workload": profile.workload,
        "rows": args.rows,
        "backend": args.backend,
        "implementation": profile.implementation,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(elapsed, 6),
        "timeout_seconds": getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        "profile_scope": getattr(args, "profile_scope", None) or profile.implementation,
        "cpu_only": _cpu_only(args),
        "cpu_sampling_rate": _cpu_sampling_rate(args),
        "json_path": "" if reduced_only else str(profile.json_path),
        "reduced_json_path": reduced_json_path,
        "html_path": "" if reduced_only else str(profile.html_path),
        "run_command": " ".join(profile.run_command),
        "html_command": "" if reduced_only else " ".join(profile.html_command),
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
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-workload Scalene run timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--profile-scope",
        help="Comma-separated Scalene --profile-only scope. Defaults to the implementation name.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Include Scalene memory profiling. Reduced profiles default to CPU-only to avoid profiler hangs.",
    )
    parser.add_argument(
        "--cpu-sampling-rate",
        type=float,
        help=(
            "Scalene CPU sampling interval in seconds. "
            f"Reduced CPU-only profiles default to {DEFAULT_REDUCED_CPU_SAMPLING_RATE:g}s."
        ),
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Generate a reduced profile (only high CPU/memory lines).",
    )
    parser.add_argument(
        "--reduced-top",
        type=int,
        default=20,
        help="Number of top entries to keep when generating reduced profiles.",
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
        help="Print Scalene commands and write manifests without launching Scalene.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for implementation in _implementations(args.implementation):
            for name, workload in workload_registry(implementation).items():
                print(f"{name}\t{workload.dataset}\t{implementation}\t{workload.description}")
        return 0
    return run_profiles(args)


def _cpu_only(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "reduced", False) and not getattr(args, "profile_memory", False))


def _cpu_sampling_rate(args: argparse.Namespace) -> float | None:
    sampling_rate = getattr(args, "cpu_sampling_rate", None)
    if sampling_rate is not None:
        return sampling_rate
    if _cpu_only(args):
        return DEFAULT_REDUCED_CPU_SAMPLING_RATE
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
