"""Shared helpers for Brightkite profiling runners."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from tests.profiling.brightkite_workloads import IMPLEMENTATIONS, _skmob_workload_candidates, workload_registry

WorkloadSelector = Callable[[list[str] | None, str], list[str]]


def implementations(requested: str) -> list[str]:
    if requested == "both":
        return list(IMPLEMENTATIONS)
    return [requested]


def profile_jobs(
    requested_implementation: str,
    requested_workloads: list[str] | None,
    selector: WorkloadSelector,
) -> list[tuple[str, str]]:
    return [
        (implementation, workload)
        for implementation in implementations(requested_implementation)
        for workload in selector(requested_workloads, implementation)
    ]


def select_workloads(
    requested: list[str] | None,
    implementation: str = "fastmob",
    *,
    default_workloads: list[str] | None = None,
    allow_unavailable: bool = False,
) -> list[str]:
    registry = workload_registry(implementation)
    if allow_unavailable and implementation == "skmob":
        registry = _skmob_workload_candidates()
    if implementation == "skmob" and not registry:
        raise SystemExit(
            "No skmob workloads are available because scikit-mobility is not importable. "
            "Create it with `bash scripts/setup_benchmark_env.sh skmob`, then rerun this profile."
        )
    workloads = requested or default_workloads
    if workloads is None:
        return list(registry)
    unknown = sorted(set(workloads) - set(registry))
    if unknown:
        available = ", ".join(registry) or "none"
        raise SystemExit(f"Unknown workload(s) for {implementation}: {', '.join(unknown)}. Available: {available}")
    return list(workloads)


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


def require_executable(executable: str, package_name: str, *, install_hint: str | None = None) -> None:
    if shutil.which(executable) is None:
        hint = install_hint or f"Install the dev extras first or install {package_name!r}."
        raise SystemExit(f"{executable!r} is not installed or not on PATH. {hint}")
