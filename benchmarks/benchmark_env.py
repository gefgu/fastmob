"""CPU and Python environment detection for benchmark result organization."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_RESULTS_BASE_DIR = Path(__file__).resolve().parent / "results"

_CPU_INFO_CACHE: dict[str, Any] | None = None


def _detect_cpu_info_impl() -> dict[str, Any]:
    system = platform.system()
    try:
        if system == "Linux":
            return _detect_linux()
        if system == "Darwin":
            return _detect_macos()
        if system == "Windows":
            return _detect_windows()
    except Exception:  # noqa: BLE001, S110
        pass
    return _fallback()


def _detect_linux() -> dict[str, Any]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _fallback()

    model = ""
    for line in text.splitlines():
        if re.match(r"model name\s*:", line, re.IGNORECASE):
            model = line.split(":", 1)[1].strip()
            break
    if not model:
        for line in text.splitlines():
            if re.match(r"(hardware|model)\s*:", line, re.IGNORECASE):
                model = line.split(":", 1)[1].strip()
                break
    if not model:
        model = platform.machine() or "unknown"

    cores = text.count("\nprocessor\t:") + (1 if text.startswith("processor\t:") else 0)
    if cores == 0:
        cores = os.cpu_count() or 1

    vendor_slug = "unknown"
    for line in text.splitlines():
        if re.match(r"vendor_id\s*:", line, re.IGNORECASE):
            vid = line.split(":", 1)[1].strip().lower()
            if "intel" in vid:
                vendor_slug = "intel"
            elif "amd" in vid or "authentic" in vid:
                vendor_slug = "amd"
            break
    if vendor_slug == "unknown":
        vendor_slug = _vendor_from_model(model)

    return {"model": model, "cores": cores, "vendor_slug": vendor_slug}


def _detect_macos() -> dict[str, Any]:
    try:
        model = subprocess.run(  # noqa: PLW1510
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        cores_str = subprocess.run(  # noqa: PLW1510
            ["sysctl", "-n", "hw.logicalcpu"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return _fallback()

    if not model:
        model = platform.machine() or "unknown"
    try:
        cores = int(cores_str)
    except ValueError:
        cores = os.cpu_count() or 1

    return {"model": model, "cores": cores, "vendor_slug": _vendor_from_model(model)}


def _detect_windows() -> dict[str, Any]:
    try:
        result = subprocess.run(  # noqa: PLW1510
            ["wmic", "cpu", "get", "Name,NumberOfLogicalProcessors", "/format:csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip() and "Node" not in ln]
        if lines:
            parts = lines[0].split(",")
            model = parts[2].strip() if len(parts) > 2 else ""
            try:
                cores = int(parts[1].strip()) if len(parts) > 1 else (os.cpu_count() or 1)
            except ValueError:
                cores = os.cpu_count() or 1
            return {"model": model or "unknown", "cores": cores, "vendor_slug": _vendor_from_model(model)}
    except Exception:  # noqa: BLE001, S110
        pass
    return _fallback()


def _vendor_from_model(model: str) -> str:
    m = model.lower()
    if "apple" in m:
        return "apple"
    if "intel" in m:
        return "intel"
    if "amd" in m:
        return "amd"
    if "arm" in m or "cortex" in m or "neoverse" in m:
        return "arm"
    return "unknown"


def _fallback() -> dict[str, Any]:
    return {"model": "unknown", "cores": os.cpu_count() or 1, "vendor_slug": "unknown"}


def detect_cpu_info() -> dict[str, Any]:
    global _CPU_INFO_CACHE
    if _CPU_INFO_CACHE is not None:
        return _CPU_INFO_CACHE
    _CPU_INFO_CACHE = _detect_cpu_info_impl()
    return _CPU_INFO_CACHE


def _normalize_cpu_model(model: str) -> str:
    s = re.sub(r"\(R\)|\(TM\)|\(C\)", "", model, flags=re.IGNORECASE)
    s = re.sub(r"@\s*[\d.]+\s*GHz", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(cpu|processor|\d+-core)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = s.strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return s[:40]


def build_env_slug(
    cpu_info: dict[str, Any],
    python_version_info: tuple[int, int] | None = None,
) -> str:
    if python_version_info is None:
        python_version_info = (sys.version_info.major, sys.version_info.minor)
    major, minor = python_version_info
    normalized = _normalize_cpu_model(str(cpu_info.get("model", "unknown")))
    cores = int(cpu_info.get("cores", os.cpu_count() or 1))
    return f"py{major}{minor}_{normalized}_{cores}c"


def get_default_output_dir() -> Path:
    cpu_info = detect_cpu_info()
    slug = build_env_slug(cpu_info)
    return _RESULTS_BASE_DIR / slug
