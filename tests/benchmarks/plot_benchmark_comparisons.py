#!/usr/bin/env python3
"""Generate marketing-style benchmark comparison plots."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skmob2-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


CANVAS = "#0a0a0a"
SURFACE_CARD = "#1a1a1a"
SURFACE_SOFT = "#121212"
HAIRLINE = "#2a2a2a"
PRIMARY = "#faff69"
ON_DARK = "#ffffff"
BODY = "#cccccc"
BODY_STRONG = "#e6e6e6"
MUTED = "#888888"
MUTED_SOFT = "#5a5a5a"
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "skmob_public_api_catalog.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_label(result: dict[str, Any]) -> str:
    label = result.get("label")
    if label is not None:
        return str(label)
    return "all"


def format_size_label(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1_000_000 and number % 1_000_000 == 0:
        return f"{number // 1_000_000}M"
    if number >= 1_000 and number % 1_000 == 0:
        return f"{number // 1_000}k"
    return f"{number:,}"


def result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result_label(item): item for item in payload.get("results", [])}


def parse_size_key(label: str) -> tuple[int, float | str]:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmM]?)", label.strip())
    if not match:
        return (1, label)
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000
    return (0, number)


def common_labels(
    original: dict[str, dict[str, Any]],
    optimized: dict[str, dict[str, Any]],
    requested_sizes: list[str] | None,
) -> list[str]:
    available = sorted(set(original).intersection(optimized), key=parse_size_key)
    if not requested_sizes:
        return available
    requested = [format_size_label(size) for size in requested_sizes]
    missing = [size for size in requested if size not in available]
    for size in missing:
        print(
            f"Skipping {size}: size label not found in both files. "
            f"Original labels: {sorted(original, key=parse_size_key)}; "
            f"skmob2 labels: {sorted(optimized, key=parse_size_key)}."
        )
    return [size for size in requested if size in available]


def expected_metrics_from_catalog(catalog_path: Path, suite: str) -> list[str]:
    catalog = load_json(catalog_path)
    return sorted(entry["name"] for entry in catalog.get("entries", []) if entry.get("suite") == suite)


def metric_rows(
    original_result: dict[str, Any],
    optimized_result: dict[str, Any],
    sort_mode: str,
    expected_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    original_metrics = original_result.get("metrics", {})
    optimized_metrics = optimized_result.get("metrics", {})
    metric_names = expected_metrics or sorted(set(original_metrics).intersection(optimized_metrics))
    rows = []
    missing_reasons = missing_metric_reasons(original_metrics, optimized_metrics, metric_names)
    if missing_reasons:
        reason_lines = "; ".join(f"{metric}: {reason}" for metric, reason in missing_reasons)
        raise ValueError(f"Cannot plot incomplete benchmark result: {reason_lines}")

    for metric in metric_names:
        original_time = original_metrics[metric]["average_seconds"]
        optimized_time = optimized_metrics[metric]["average_seconds"]
        rows.append(
            {
                "metric": metric,
                "original": float(original_time),
                "optimized": float(optimized_time),
                "speedup": float(original_time) / float(optimized_time),
            }
        )

    if sort_mode == "speedup":
        rows.sort(key=lambda row: row["speedup"], reverse=True)
    else:
        rows.sort(key=lambda row: row["original"], reverse=True)
    return rows


def missing_metric_reasons(
    original_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    metric_names: list[str],
) -> list[tuple[str, str]]:
    reasons = []
    for metric in metric_names:
        original_metric = original_metrics.get(metric)
        optimized_metric = optimized_metrics.get(metric)
        if original_metric is None:
            reasons.append((metric, "missing from original skmob JSON"))
            continue
        if optimized_metric is None:
            reasons.append((metric, "missing from skmob2 JSON"))
            continue

        original_reason = invalid_metric_reason(original_metric, "original skmob")
        optimized_reason = invalid_metric_reason(optimized_metric, "skmob2")
        if original_reason:
            reasons.append((metric, original_reason))
        if optimized_reason:
            reasons.append((metric, optimized_reason))
    return reasons


def invalid_metric_reason(metric_result: dict[str, Any], label: str) -> str | None:
    status = metric_result.get("status")
    average_seconds = metric_result.get("average_seconds")
    if status not in (None, "ok"):
        reason = metric_result.get("reason")
        if reason:
            return f"{label} status={status} ({reason})"
        return f"{label} status={status}"
    if not is_valid_time(average_seconds):
        return f"{label} average_seconds is not a positive finite number"
    return None


def is_valid_time(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return value > 0 and math.isfinite(value)


def display_metric_name(metric: str) -> str:
    return metric.replace("_", " ")


def comparison_title(suite: str, backend: str | None) -> str:
    if backend:
        return f"skmob2 {backend} vs original skmob"
    return "skmob2 vs original skmob"


def comparison_subtitle(
    original_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
    suite: str,
    backend: str | None,
    size_label: str,
) -> str:
    metadata = optimized_payload.get("metadata", {})
    original_metadata = original_payload.get("metadata", {})
    iterations = metadata.get("iterations") or original_metadata.get("iterations")
    parts = [suite.title()]
    if backend:
        parts.append(f"{backend.title()} backend")
    parts.append(f"{size_label} rows")
    if size_label == "all":
        parts = [suite.title()]
        if backend:
            parts.append(f"{backend.title()} backend")
    if iterations:
        parts.append(f"{iterations} iterations")
    return " / ".join(parts)


def comparison_context(original_payload: dict[str, Any], optimized_payload: dict[str, Any]) -> str:
    original_metadata = original_payload.get("metadata", {})
    optimized_metadata = optimized_payload.get("metadata", {})
    original_input = original_metadata.get("input_type") or "original input"
    optimized_input = optimized_metadata.get("input_type") or "skmob2 input"
    timing_mode = str(original_metadata.get("timing_mode") or "")
    if timing_mode == "prebuilt_tdf":
        baseline_note = "Original skmob uses a prebuilt TrajDataFrame baseline"
    elif timing_mode:
        baseline_note = f"Original skmob timing mode: {timing_mode.replace('_', ' ')}"
    else:
        baseline_note = "Original skmob baseline"
    return f"{baseline_note} / {original_input} vs {optimized_input}"


def output_name(suite: str, backend: str | None, size_label: str) -> str:
    safe_size = re.sub(r"[^A-Za-z0-9_.-]+", "_", size_label)
    if not backend:
        return f"skmob2_vs_skmob_{suite}_{safe_size}.png"
    return f"skmob2_vs_skmob_{suite}_{backend}_{safe_size}.png"


def add_round_card(fig: plt.Figure) -> None:
    card = FancyBboxPatch(
        (0.025, 0.035),
        0.95,
        0.93,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor=SURFACE_CARD,
        edgecolor=HAIRLINE,
        linewidth=1.0,
        zorder=-1,
    )
    fig.patches.append(card)


def draw_plot(
    rows: list[dict[str, Any]],
    *,
    original_payload: dict[str, Any],
    optimized_payload: dict[str, Any],
    suite: str,
    backend: str | None,
    size_label: str,
    output_path: Path,
) -> None:
    metric_count = len(rows)
    figure_height = max(7.4, min(12.8, 2.8 + metric_count * 0.70))
    fig, ax = plt.subplots(figsize=(14.0, figure_height), facecolor=CANVAS)
    add_round_card(fig)
    plot_left = 0.27
    plot_right = 0.92
    header_left = plot_left
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.68, bottom=0.12)
    ax.set_facecolor(SURFACE_CARD)

    top_speedup = max(row["speedup"] for row in rows)
    median_speedup = sorted(row["speedup"] for row in rows)[metric_count // 2]
    max_time = max(row["original"] for row in rows)
    x_max = max_time * 1.22
    group_gap = 1.98
    y_positions = [index * group_gap for index in range(metric_count)]
    pair_offset = 0.46
    bar_height = 0.64

    original_values = [row["original"] for row in rows]
    optimized_values = [row["optimized"] for row in rows]
    labels = [display_metric_name(row["metric"]) for row in rows]

    ax.barh(
        [y - pair_offset for y in y_positions],
        original_values,
        height=bar_height,
        color=MUTED_SOFT,
        edgecolor=MUTED,
        linewidth=0.6,
        label="Original skmob",
    )
    ax.barh(
        [y + pair_offset for y in y_positions],
        optimized_values,
        height=bar_height,
        color=PRIMARY,
        edgecolor=PRIMARY,
        linewidth=0.6,
        label=f"skmob2 {backend}" if backend else "skmob2",
    )

    ax.set_xlim(0, x_max)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=BODY_STRONG, fontsize=16)
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors=MUTED, labelsize=15)
    ax.tick_params(axis="y", colors=BODY_STRONG)
    ax.xaxis.grid(True, color=HAIRLINE, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average execution time (seconds)", color=BODY, fontsize=17, labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    offset = x_max * 0.012
    optimized_label_floor = x_max * 0.035
    multiplier_x = x_max * 0.965
    leader_end_x = x_max * 0.90
    for index, (group_y, row) in enumerate(zip(y_positions, rows, strict=True)):
        original_y = group_y - pair_offset
        optimized_y = group_y + pair_offset
        original_label_inside = row["original"] >= x_max * 0.06
        original_label_x = row["original"] - offset if original_label_inside else row["original"] + offset
        ax.text(
            original_label_x,
            original_y,
            format_seconds(row["original"]),
            va="center",
            ha="right" if original_label_inside else "left",
            color=ON_DARK,
            fontsize=12.5,
            fontweight="bold",
        )

        optimized_label_x = row["optimized"] + offset
        needs_callout = optimized_label_x < optimized_label_floor
        if needs_callout:
            optimized_label_x = optimized_label_floor
            ax.plot(
                [row["optimized"], optimized_label_x - offset * 0.35],
                [optimized_y, optimized_y],
                color=PRIMARY,
                alpha=0.42,
                linewidth=0.8,
                solid_capstyle="round",
                zorder=4,
            )
        ax.text(
            min(optimized_label_x, x_max * 0.985),
            optimized_y,
            format_seconds(row["optimized"]),
            va="center",
            ha="left",
            color=PRIMARY,
            fontsize=12.5,
            fontweight="bold",
        )
        leader_start_x = min(max(row["original"], row["optimized"]) + offset * 1.2, leader_end_x)
        ax.plot(
            [leader_start_x, leader_end_x],
            [group_y, group_y],
            color=PRIMARY,
            alpha=0.22,
            linewidth=0.8,
            linestyle=(0, (1.5, 4.0)),
            zorder=1,
        )
        ax.text(
            multiplier_x,
            group_y,
            f"{row['speedup']:.2f}x",
            va="center",
            ha="right",
            color=CANVAS,
            fontsize=13.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.14",
                "facecolor": PRIMARY,
                "edgecolor": PRIMARY,
                "linewidth": 0,
            },
        )

    fig.text(
        header_left,
        0.965,
        comparison_title(suite, backend),
        color=ON_DARK,
        fontsize=25,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.915,
        comparison_subtitle(original_payload, optimized_payload, suite, backend, size_label),
        color=BODY,
        fontsize=13,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.883,
        f"Median {median_speedup:.2f}x across {metric_count} shared benchmarks / lower is better",
        color=MUTED,
        fontsize=11.5,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.855,
        comparison_context(original_payload, optimized_payload),
        color=MUTED,
        fontsize=10.5,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.807,
        f"up to {top_speedup:.2f}x faster",
        color=PRIMARY,
        fontsize=38,
        fontweight="bold",
        ha="left",
        va="top",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=((plot_left + plot_right) / 2, 0.735),
        ncols=2,
        frameon=False,
        fontsize=14,
        borderpad=0.45,
        labelspacing=0.45,
        columnspacing=1.35,
        handlelength=2.0,
        handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=CANVAS, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def format_seconds(value: float) -> str:
    if value < 0.01:
        return f"{value * 1_000:.2f} ms"
    if value < 1:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def generate_plots(args: argparse.Namespace) -> int:
    original_path = args.original_json.resolve()
    optimized_path = args.optimized_json.resolve()
    if not original_path.exists():
        print(f"ERROR: original benchmark JSON not found: {original_path}")
        return 2
    if not optimized_path.exists():
        print(f"ERROR: optimized benchmark JSON not found: {optimized_path}")
        return 2

    original_payload = load_json(original_path)
    optimized_payload = load_json(optimized_path)
    original_results = result_map(original_payload)
    optimized_results = result_map(optimized_payload)
    labels = common_labels(original_results, optimized_results, args.sizes)
    if not labels:
        print(
            "No overlapping size labels found. "
            f"Original labels: {sorted(original_results, key=parse_size_key)}; "
            f"skmob2 labels: {sorted(optimized_results, key=parse_size_key)}."
        )
        if args.sizes:
            return 0
        return 1

    expected_metrics = expected_metrics_from_catalog(args.catalog, args.suite)
    if not expected_metrics:
        print(f"ERROR: no expected metrics for suite '{args.suite}' in {args.catalog}.")
        return 2

    generated = 0
    for label in labels:
        try:
            rows = metric_rows(original_results[label], optimized_results[label], args.sort, expected_metrics)
        except ValueError as exc:
            print(f"ERROR for {label}: {exc}")
            return 1
        if not rows:
            print(f"No valid overlapping metrics found for {label}; skipping.")
            continue
        output_path = args.output_dir / output_name(args.suite, args.backend, label)
        draw_plot(
            rows,
            original_payload=original_payload,
            optimized_payload=optimized_payload,
            suite=args.suite,
            backend=args.backend,
            size_label=label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    if generated == 0:
        print("No plots were generated.")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-json", required=True, type=Path, help="Original skmob benchmark JSON.")
    parser.add_argument("--optimized-json", required=True, type=Path, help="skmob2 benchmark JSON.")
    parser.add_argument("--suite", required=True, help="Benchmark suite name, for example spatial or privacy.")
    parser.add_argument("--backend", choices=("pandas", "polars"), help="skmob2 backend.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Public API catalog used to require complete suite metric coverage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/benchmarks/results/plots"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument("--sizes", nargs="*", help="Optional size labels such as 1M 4M.")
    parser.add_argument(
        "--sort",
        choices=("speedup", "original-time"),
        default="speedup",
        help="Metric ordering for the plot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    return generate_plots(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
