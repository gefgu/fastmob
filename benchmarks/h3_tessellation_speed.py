"""Compare Python-H3 and native Rust H3 polygon tessellation.

Python H3 is used only as an optional benchmark baseline; it is not a Fastmob
runtime dependency.  The workload includes both polygon coverage and cell
boundary extraction, matching ``H3TessellationTiler``.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

from fastmob._core import h3_cells_to_boundaries, h3_polygons_to_cells

POLYGON = [
    [
        (-0.10, 51.50),
        (0.10, 51.50),
        (0.10, 51.60),
        (-0.10, 51.60),
        (-0.10, 51.50),
    ]
]


def legacy_h3(resolution: int):
    import h3.api.numpy_int as h3

    geometry = {"type": "Polygon", "coordinates": POLYGON}
    cells = list(h3.geo_to_cells(geometry, resolution))
    return cells, [h3.cell_to_boundary(cell) for cell in cells]


def native_h3(resolution: int):
    cells = h3_polygons_to_cells([POLYGON], resolution)
    return cells, h3_cells_to_boundaries(cells)


def measure(func, resolution: int, repeats: int):
    durations, peaks = [], []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        cells, boundaries = func(resolution)
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(cells) == len(boundaries) > 0
        durations.append(elapsed)
        peaks.append(peak / (1024 * 1024))
    return statistics.median(durations), statistics.mean(peaks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[7, 9])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        import h3.api.numpy_int  # noqa: F401
    except ImportError:
        print("Python H3 is not installed; native benchmark baseline skipped.")
        return 0

    for resolution in args.resolutions:
        legacy_h3(resolution)
        native_h3(resolution)
        legacy_s, legacy_mb = measure(legacy_h3, resolution, args.repeats)
        native_s, native_mb = measure(native_h3, resolution, args.repeats)
        print(
            f"resolution={resolution}: legacy={legacy_s:.4f}s/{legacy_mb:.2f}MiB, "
            f"native={native_s:.4f}s/{native_mb:.2f}MiB, speedup={legacy_s / native_s:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
