from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


METRICS = {
    "total_cpu_percent": ("n_cpu_percent_python", "n_cpu_percent_c", "n_sys_percent"),
    "python_cpu_percent": ("n_cpu_percent_python",),
    "native_cpu_percent": ("n_cpu_percent_c",),
    "system_cpu_percent": ("n_sys_percent",),
    "peak_memory_mb": ("n_peak_mb",),
    "allocated_mb": ("n_malloc_mb",),
    "allocation_count": ("n_mallocs",),
    "growth_mb": ("n_growth_mb",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce Scalene JSON to the hottest files, functions, and lines.")
    parser.add_argument("profile_json", type=Path, help="Path to a Scalene JSON profile.")
    parser.add_argument("--top", type=positive_int, default=20, help="Maximum entries per ranking.")
    parser.add_argument("-o", "--output", type=Path, help="Write reduced JSON to this path instead of stdout.")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def metric_value(entry: dict[str, Any], fields: tuple[str, ...]) -> float:
    if len(fields) == 1 and fields[0] not in entry:
        summary_key = {
            "n_cpu_percent_python": "python_cpu_percent",
            "n_cpu_percent_c": "native_cpu_percent",
            "n_sys_percent": "system_cpu_percent",
            "n_peak_mb": "peak_memory_mb",
            "n_malloc_mb": "allocated_mb",
            "n_mallocs": "allocation_count",
            "n_growth_mb": "growth_mb",
        }.get(fields[0])
        if summary_key is not None:
            return float(entry.get(summary_key) or 0)
    if fields == METRICS["total_cpu_percent"] and fields[0] not in entry:
        return float(entry.get("total_cpu_percent") or 0)
    return sum(float(entry.get(field) or 0) for field in fields)


def summarize_entry(file_path: str, entry: dict[str, Any], *, kind: str) -> dict[str, Any]:
    summary = {
        "kind": kind,
        "file": file_path,
        "line": entry.get("lineno"),
        "code": (entry.get("line") or "").strip(),
        "python_cpu_percent": round(metric_value(entry, METRICS["python_cpu_percent"]), 6),
        "native_cpu_percent": round(metric_value(entry, METRICS["native_cpu_percent"]), 6),
        "system_cpu_percent": round(metric_value(entry, METRICS["system_cpu_percent"]), 6),
        "total_cpu_percent": round(metric_value(entry, METRICS["total_cpu_percent"]), 6),
        "peak_memory_mb": round(metric_value(entry, METRICS["peak_memory_mb"]), 6),
        "allocated_mb": round(metric_value(entry, METRICS["allocated_mb"]), 6),
        "allocation_count": int(metric_value(entry, METRICS["allocation_count"])),
        "growth_mb": round(metric_value(entry, METRICS["growth_mb"]), 6),
    }
    if "start_function_line" in entry:
        summary["function_start_line"] = entry.get("start_function_line")
    if "end_function_line" in entry:
        summary["function_end_line"] = entry.get("end_function_line")
    return summary


def top_by(entries: list[dict[str, Any]], metric: str, top: int) -> list[dict[str, Any]]:
    fields = METRICS[metric]
    ranked = sorted(entries, key=lambda entry: metric_value(entry, fields), reverse=True)
    return [entry for entry in ranked[:top] if metric_value(entry, fields) > 0]


def reduce_profile(payload: dict[str, Any], *, top: int) -> dict[str, Any]:
    files = payload.get("files") or {}
    line_entries: list[dict[str, Any]] = []
    function_entries: list[dict[str, Any]] = []
    leak_entries: list[dict[str, Any]] = []

    top_files = []
    for file_path, detail in files.items():
        lines = detail.get("lines") or []
        functions = detail.get("functions") or []
        top_files.append(
            {
                "file": file_path,
                "percent_cpu_time": round(float(detail.get("percent_cpu_time") or 0), 6),
                "line_count": len(lines),
                "function_count": len(functions),
            }
        )
        for line in lines:
            line_entries.append(summarize_entry(file_path, line, kind="line"))
        for function in functions:
            function_entries.append(summarize_entry(file_path, function, kind="function"))
        for line_no, leak in (detail.get("leaks") or {}).items():
            leak_entries.append(
                {
                    "file": file_path,
                    "line": int(line_no) if str(line_no).isdigit() else line_no,
                    "likelihood": leak.get("likelihood"),
                    "velocity_mb_s": leak.get("velocity_mb_s"),
                }
            )

    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {"lines": {}, "functions": {}}
    for metric in METRICS:
        rankings["lines"][metric] = top_by(line_entries, metric, top)
        rankings["functions"][metric] = top_by(function_entries, metric, top)

    return {
        "metadata": {
            "program": payload.get("program"),
            "filename": payload.get("filename"),
            "args": payload.get("args"),
            "elapsed_time_sec": payload.get("elapsed_time_sec"),
            "memory": payload.get("memory"),
            "gpu": payload.get("gpu"),
            "max_footprint_mb": payload.get("max_footprint_mb"),
            "max_footprint_location": {
                "file": payload.get("max_footprint_fname"),
                "line": payload.get("max_footprint_lineno"),
                "python_fraction": payload.get("max_footprint_python_fraction"),
            },
        },
        "top_files_by_cpu": sorted(top_files, key=lambda item: item["percent_cpu_time"], reverse=True)[:top],
        "rankings": rankings,
        "leaks": sorted(leak_entries, key=lambda item: item.get("likelihood") or 0, reverse=True)[:top],
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.profile_json.read_text(encoding="utf-8"))
    reduced = reduce_profile(payload, top=args.top)
    text = json.dumps(reduced, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
