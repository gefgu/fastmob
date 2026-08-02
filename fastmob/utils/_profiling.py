"""Small opt-in profiling helpers for Python API wrappers."""

from __future__ import annotations

import os
import sys
import time


class StageTimer:
    """Collect and emit named wall-clock stages when an environment flag is set."""

    def __init__(self, environment_variable: str, label: str) -> None:
        self._enabled = os.environ.get(environment_variable) is not None
        self._label = label
        self._started = time.perf_counter()
        self._stage_started = self._started
        self._timings: dict[str, float] = {}

    def checkpoint(self, name: str) -> None:
        """Record elapsed time since the preceding checkpoint when enabled."""
        if not self._enabled:
            return
        now = time.perf_counter()
        self._timings[name] = now - self._stage_started
        self._stage_started = now

    def emit(self) -> None:
        """Write the collected timings to stderr when profiling is enabled."""
        if not self._enabled:
            return
        detail = " ".join(f"{name}={value:.6f}s" for name, value in self._timings.items())
        print(
            f"{self._label}: {detail} total={time.perf_counter() - self._started:.6f}s",
            file=sys.stderr,
            flush=True,
        )
