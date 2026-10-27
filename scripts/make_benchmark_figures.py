"""Regenerate every benchmark figure of the docs from the raw benchmark results.

One chart type (horizontal grouped bars, log axis) is drawn from tidy ``Run``
records, so adding a figure means adding one entry to ``FIGURES``.

Sources live in the sibling ``fastmob_benchmarks`` repository (``--bench-repo``):
the measures and preprocessing suites write ``*_benchmarks.json`` with one record
per (library, operation); the jump_lengths suite writes a hyperfine CSV.

Usage: uv run --no-project --with matplotlib python scripts/make_benchmark_figures.py
       [--bench-repo ../fastmob_benchmarks] [--out docs/src/assets/benchmarks] [name ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = "benchmarks/{suite}/results/{machine}"
I7 = "py312_intel_core_i7_10700_16c"
XEON = "py313_intel_xeon_silver_4316_40c"

# Library display names and colours, in legend order. Fastkit-Mobility is gold
# (two steps for its two dataframe backends); every other library is a muted hue.
LIBRARIES = {
    "fastmob-pandas": ("Fastkit-Mobility (pandas)", "#a98600"),
    "fastmob-polars": ("Fastkit-Mobility (Polars)", "#e3b800"),
    "skmob": ("scikit-mobility", "#8a8a8a"),
    "movingpandas": ("MovingPandas", "#4f78a8"),
    "ptrail": ("PTRAIL", "#8c6bb1"),
    "movetk": ("MoveTK", "#3f9b8a"),
    "trackintel": ("trackintel", "#b8693d"),
    "transbigdata": ("TransBigData", "#6a8f3c"),
    "humobi": ("HuMobi", "#a05a7a"),
    "tracktable": ("Tracktable", "#7a7a52"),
    "sparkmobility": ("SparkMobility", "#c0504d"),
}
CSV_NAMES = {"scikitmobility": "skmob"}
COLLECTIVE = {
    "homes_per_location",
    "visits_per_location",
    "visits_per_time_unit",
    "mean_square_displacement",
    "random_location_entropy",
    "uncorrelated_location_entropy",
}
TEXT = "#666666"
MEMORY_FLOOR_MB = 0.1


@dataclass(frozen=True)
class Run:
    op: str
    library: str  # key of LIBRARIES
    seconds: float | None
    memory_mb: float | None
    status: str  # ok | timeout | error | ...


def _key(library: str, backend: str | None) -> str:
    library = library.lower()
    return f"{library}-{backend}" if library == "fastmob" else CSV_NAMES.get(library, library)


def load_json(path: Path, op_field: str, order: str | None = None) -> list[Run]:
    results = json.loads(path.read_text())["results"]
    runs = []
    for r in results:
        if r.get("label") not in (None, "4M") or r.get("library") is None:
            continue
        if order and r.get("order") not in (None, order):
            continue
        runs.append(
            Run(
                r[op_field],
                _key(r["library"], r.get("backend")),
                r.get("average_seconds"),
                r.get("operation_memory_mb"),
                r["status"],
            )
        )
    return runs


def load_privacy_speed(folder: Path) -> list[Run]:
    """The privacy suite writes one speed JSON per library/backend, each with a ``metrics`` dict."""
    files = {
        "fastmob-pandas": "fastmob_privacy_speed_pandas.json",
        "fastmob-polars": "fastmob_privacy_speed_polars.json",
        "skmob": "skmob_privacy_speed_prebuilt_tdf.json",
    }
    runs = []
    for key, name in files.items():
        for metric, m in json.loads((folder / name).read_text())["results"][0]["metrics"].items():
            runs.append(Run(metric, key, m.get("average_seconds"), None, m["status"]))
    return runs


def load_hyperfine_csv(path: Path, op: str, size: int) -> list[Run]:
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["size"]) == size]
    return [Run(op, CSV_NAMES.get(r["library"], r["library"]), float(r["adjusted_mean_s"]), None, "ok") for r in rows]


@dataclass(frozen=True)
class Figure:
    name: str
    title: str
    subtitle: str
    runs: list[Run]
    metric: str = "seconds"  # seconds | memory_mb
    ops: list[str] | None = None  # default: every op with a non-Fastkit run, in data order


def figures(repo: Path) -> list[Figure]:
    def res(suite: str, machine: str, name: str) -> Path:
        return repo / RESULTS.format(suite=suite, machine=machine) / name

    measures = load_json(res("measures", f"{I7}_brightkite4M", "measures_benchmarks.json"), "name")
    preprocessing = load_json(res("preprocessing", XEON, "preprocessing_benchmarks.json"), "operation")
    spark = load_json(
        res("sparkmobility", XEON.replace("py313", "py312"), "sparkmobility_benchmarks.json"), "name", "raw"
    )
    jump = load_hyperfine_csv(res("jump_lengths", I7, "jump_lengths_benchmark.csv"), "jump_lengths", 4_000_000)
    # The jump_lengths CSV cites Fastkit-Mobility; use the measured values from the measures suite instead.
    jump = [r for r in jump if not r.library.startswith("fastmob")] + [
        r for r in measures if r.op == "jump_lengths" and r.library.startswith("fastmob")
    ]

    brightkite = "4M Brightkite check-ins (time-sorted input), Intel Core i7-10700"
    geolife = "4M GeoLife points, Intel Xeon Silver 4316"
    individual = [r for r in measures if r.op not in COLLECTIVE]
    collective = [r for r in measures if r.op in COLLECTIVE]
    out = []
    for metric, suffix, word in (("seconds", "speed", "Execution time"), ("memory_mb", "memory", "Peak extra memory")):
        out += [
            Figure(
                f"measures-individual-{suffix}", f"Individual measures: {word.lower()}", brightkite, individual, metric
            ),
            Figure(
                f"measures-collective-{suffix}", f"Collective measures: {word.lower()}", brightkite, collective, metric
            ),
            Figure(f"preprocessing-{suffix}", f"Preprocessing: {word.lower()}", geolife, preprocessing, metric),
        ]
    spark_where = "full GeoLife (24.9M points), Intel Xeon Silver 4316, 40 Spark cores"
    for metric, suffix, word in (("seconds", "speed", "Execution time"), ("memory_mb", "memory", "Peak extra memory")):
        out.append(Figure(f"sparkmobility-{suffix}", f"vs SparkMobility: {word.lower()}", spark_where, spark, metric))
    privacy = load_privacy_speed(res("privacy", I7, "").parent / I7)
    out.append(
        Figure(
            "privacy-speed",
            "Privacy risk: execution time",
            "toy trajectories repeated 30x, Intel Core i7-10700",
            privacy,
        )
    )
    out.append(Figure("jump-lengths-libraries", "jump_lengths across eight libraries", brightkite, jump))
    return out


def _label(run: Run, metric: str, best_other: float | None) -> str:
    value = run.seconds if metric == "seconds" else run.memory_mb
    unit = "s" if metric == "seconds" else " MB"
    if value is None:
        return run.status
    text = (
        f"{value:,.0f}{unit}"
        if metric == "seconds" and value >= 100
        else f"{value:.3g}{unit}"
        if value >= MEMORY_FLOOR_MB or metric == "seconds"
        else f"<{MEMORY_FLOOR_MB:g}{unit}"
    )
    if (
        best_other
        and run.library.startswith("fastmob")
        and (metric == "seconds" or value >= MEMORY_FLOOR_MB)
        and best_other / value >= 2
    ):
        text += (
            f"  ({best_other / value:,.0f}x less)"
            if metric == "memory_mb"
            else f"  ({best_other / value:,.0f}x faster)"
        )
    return text


def draw(fig_spec: Figure, out: Path) -> None:
    metric = fig_spec.metric
    runs = [r for r in fig_spec.runs if r.status == "ok" or r.status in ("timeout", "error")]
    by_op: dict[str, dict[str, Run]] = {}
    for r in runs:
        by_op.setdefault(r.op, {})[r.library] = r

    # Keep operations where Fastkit-Mobility and at least one other library produced a number.
    def value(r: Run | None) -> float | None:
        return None if r is None else (r.seconds if metric == "seconds" else r.memory_mb)

    ops = [
        op
        for op, libs in by_op.items()
        if any(value(r) is not None for k, r in libs.items() if k.startswith("fastmob"))
        and any(value(r) is not None for k, r in libs.items() if not k.startswith("fastmob"))
    ]
    ops.sort(key=lambda o: -max(value(r) or 0 for r in by_op[o].values()))
    present = [k for k in LIBRARIES if any(k in by_op[o] for o in ops)]
    if not ops:
        raise SystemExit(f"{fig_spec.name}: nothing to plot")

    # One unit per bar, a gap between operations; y grows downward through ``ops``.
    gap = 0.9
    fig_units = sum(len(by_op[o]) for o in ops) + gap * (len(ops) - 1)
    fig, ax = plt.subplots(figsize=(9, 0.205 * fig_units + 2.0), dpi=160)
    values = [v for o in ops for r in by_op[o].values() if (v := value(r)) is not None and v > 0]
    floor = MEMORY_FLOOR_MB if metric == "memory_mb" else min(values) / 2.5
    y, centers = 0.0, []
    for op in ops:
        runs_here = [(lib, by_op[op][lib]) for lib in present if lib in by_op[op]]
        others = [value(r) for lib, r in runs_here if not lib.startswith("fastmob") and value(r)]
        best_other = min(others) if others else None
        start = y
        for lib, r in runs_here:
            v = value(r)
            if v is None:
                ax.text(floor * 1.15, y, r.status, va="center", fontsize=7, color=TEXT, style="italic")
            else:
                ax.barh(y, max(v, floor) - floor, left=floor, height=0.82, color=LIBRARIES[lib][1])
                ax.text(max(v, floor) * 1.2, y, _label(r, metric, best_other), va="center", fontsize=7, color=TEXT)
            y += 1
        centers.append((start + y - 1) / 2)
        y += gap
    ax.set_xscale("log")
    ax.set_yticks(centers, [o.replace("_", " ") for o in ops], fontsize=9, color=TEXT)
    ax.set_ylim(y - gap - 0.5, -0.5)
    ax.set_xlim(floor, max(values) * 14)
    ax.tick_params(axis="x", colors=TEXT, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        f"{'seconds' if metric == 'seconds' else 'megabytes'}, log scale, lower is better", fontsize=9, color=TEXT
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=LIBRARIES[lib][1]) for lib in present],
        labels=[LIBRARIES[lib][0] for lib in present],
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    fig.text(0.01, 0.995, fig_spec.title, ha="left", va="top", fontsize=12, color=TEXT, fontweight="bold")
    fig.text(0.01, 0.995 - 0.3 / fig.get_figheight(), fig_spec.subtitle, ha="left", va="top", fontsize=8.5, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out / f"{fig_spec.name}.png", transparent=True)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="figure names to draw (default: all)")
    ap.add_argument("--bench-repo", type=Path, default=Path("../fastmob_benchmarks"))
    ap.add_argument("--out", type=Path, default=Path("docs/src/assets/benchmarks"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in figures(args.bench_repo):
        if not args.names or spec.name in args.names:
            draw(spec, args.out)
            print(f"wrote {args.out / (spec.name + '.png')}")


if __name__ == "__main__":
    main()
