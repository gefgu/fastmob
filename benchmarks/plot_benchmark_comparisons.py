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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter"]


CANVAS = "#0a0a0a"
SURFACE_CARD = "#1a1a1a"
SURFACE_SOFT = "#121212"
HAIRLINE = "#2a2a2a"
PRIMARY = "#faff69"
NEGATIVE_COLOR = "#ff6b6b"
ON_DARK = "#ffffff"
BODY = "#cccccc"
BODY_STRONG = "#e6e6e6"
MUTED = "#888888"
MUTED_SOFT = "#5a5a5a"
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "skmob_public_api_catalog.json"
MODEL_TRAJECTORY_METRICS = ["epr", "density_epr", "spatial_epr", "geosim", "sts_epr"]
MODEL_LOCATION_METRICS = ["gravity_flows", "gravity_probabilities", "radiation_flows", "radiation_probabilities"]
_LARGE_SPEC_RE = re.compile(r"^(.*?)_(\d+)a$")

FIGURE_WIDTH = 11.0
FIGURE_MIN_HEIGHT = 7.4
FIGURE_MAX_HEIGHT = 22
FIGURE_BASE_HEIGHT = 2.8
FIGURE_HEIGHT_PER_METRIC = 0.70

TITLE_FONT_SIZE = 36
SUBTITLE_FONT_SIZE = 15
DETAIL_FONT_SIZE = 12.5
CONTEXT_FONT_SIZE = 11.5
CALLOUT_FONT_SIZE = 42
LEGEND_FONT_SIZE = 16
METRIC_LABEL_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 20
BAR_LABEL_FONT_SIZE = 12.5
SPEEDUP_VALUE_FONT_SIZE = 13.5
SPEEDUP_HEADER_FONT_SIZE = 11
FOOTER_FONT_SIZE = 9.5

BAR_HEIGHT = 1
BAR_PAIR_OFFSET = 0.6
BAR_GROUP_GAP = 3


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_env_footer(optimized_payload: dict[str, Any]) -> str | None:
    """Return 'Python 3.12  ·  Intel Core i7-12700K  ·  12 logical cores' or None."""
    metadata = optimized_payload.get("metadata", {})
    cpu_model = metadata.get("cpu_model")
    if not cpu_model:
        return None
    parts: list[str] = []
    py_ver = metadata.get("python_version", "")
    m = re.match(r"(\d+\.\d+)", py_ver)
    if m:
        parts.append(f"Python {m.group(1)}")
    parts.append(cpu_model)
    cores = metadata.get("cpu_cores")
    if cores:
        parts.append(f"{cores} logical cores")
    return "  ·  ".join(parts) if parts else None


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


def model_trajectory_results(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    for item in payload.get("results", []):
        if item.get("benchmark_group") != "trajectory_models":
            continue
        n_agents = item.get("n_agents")
        n_locations = item.get("n_locations")
        if isinstance(n_agents, int) and isinstance(n_locations, int):
            rows[(n_locations, n_agents)] = item
    return rows


def requested_location_counts(requested_sizes: list[str] | None) -> list[int] | None:
    if not requested_sizes:
        return None
    counts = []
    for size in requested_sizes:
        parsed = parse_size_key(size)
        if parsed[0] == 0:
            counts.append(int(parsed[1]))
    return counts


def common_model_locations(
    original: dict[tuple[int, int], dict[str, Any]],
    optimized: dict[tuple[int, int], dict[str, Any]],
    requested_sizes: list[str] | None,
) -> list[int]:
    original_locations = {n_locations for n_locations, _ in original}
    optimized_locations = {n_locations for n_locations, _ in optimized}
    available = sorted(original_locations.intersection(optimized_locations))
    requested = requested_location_counts(requested_sizes)
    if requested is None:
        return available
    missing = [count for count in requested if count not in available]
    for count in missing:
        print(
            f"Skipping {count} locations: location count not found in both model files. "
            f"Original locations: {original_locations}; skmob2 locations: {optimized_locations}."
        )
    return [count for count in requested if count in available]


def model_matrix_rows(
    original_results: dict[tuple[int, int], dict[str, Any]],
    optimized_results: dict[tuple[int, int], dict[str, Any]],
    n_locations: int,
    sort_mode: str,
    metric_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    metric_names = metric_names or MODEL_TRAJECTORY_METRICS
    original_agents = {n_agents for locs, n_agents in original_results if locs == n_locations}
    optimized_agents = {n_agents for locs, n_agents in optimized_results if locs == n_locations}
    agent_counts = sorted(original_agents.intersection(optimized_agents))
    rows = []
    missing_reasons = []

    for n_agents in agent_counts:
        original_result = original_results[(n_locations, n_agents)]
        optimized_result = optimized_results[(n_locations, n_agents)]
        for metric in metric_names:
            reasons = missing_metric_reasons(
                original_result.get("metrics", {}),
                optimized_result.get("metrics", {}),
                [metric],
            )
            if reasons:
                missing_reasons.extend((f"{metric} / {n_agents} agents", reason) for _, reason in reasons)
                continue
            original_time = original_result["metrics"][metric]["average_seconds"]
            optimized_time = optimized_result["metrics"][metric]["average_seconds"]
            rows.append(
                {
                    "metric": f"{metric} / {n_agents} agents",
                    "original": float(original_time),
                    "optimized": float(optimized_time),
                    "speedup": float(original_time) / float(optimized_time),
                }
            )

    if missing_reasons:
        reason_lines = "; ".join(f"{metric}: {reason}" for metric, reason in missing_reasons)
        raise ValueError(f"Cannot plot incomplete benchmark result: {reason_lines}")

    if sort_mode == "speedup":
        rows.sort(key=lambda row: row["speedup"], reverse=True)
    else:
        rows.sort(key=lambda row: row["original"], reverse=True)
    return rows


def expected_metrics_from_catalog(catalog_path: Path, suite: str) -> list[str]:
    catalog = load_json(catalog_path)
    return sorted(entry["name"] for entry in catalog.get("entries", []) if entry.get("suite") == suite)


def metric_rows(
    original_result: dict[str, Any],
    optimized_result: dict[str, Any],
    sort_mode: str,
    expected_metrics: list[str] | None = None,
    *,
    skip_invalid: bool = False,
) -> list[dict[str, Any]]:
    original_metrics = original_result.get("metrics", {})
    optimized_metrics = optimized_result.get("metrics", {})
    metric_names = expected_metrics or sorted(set(original_metrics).intersection(optimized_metrics))
    rows = []
    missing_reasons = missing_metric_reasons(original_metrics, optimized_metrics, metric_names)
    if missing_reasons:
        if skip_invalid:
            invalid_metrics = {metric for metric, _ in missing_reasons}
            metric_names = [metric for metric in metric_names if metric not in invalid_metrics]
        else:
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
    return metric.replace("_", " ").replace("kl2", "")


def comparison_title(suite: str, backend: str | None) -> str:
    return "skmob2 vs skmob"


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
    if suite == "models":
        parts.append(size_label)
    else:
        parts.append(f"{size_label} rows")
    if size_label == "all":
        parts = [suite.title()]
        if backend:
            parts.append(f"{backend.title()} backend")
    if iterations:
        parts.append(f"{iterations} iterations")
    if metadata.get("input_order") == "sorted" or original_metadata.get("input_order") == "sorted":
        parts.append("sorted")
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


def _resolve_output_dir(base: Path, *payloads: dict[str, Any]) -> Path:
    """Return base/sorted/ when any payload was generated from sorted input."""
    if any(p.get("metadata", {}).get("input_order") == "sorted" for p in payloads):
        return base / "sorted"
    return base


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
    figure_height = max(
        FIGURE_MIN_HEIGHT,
        min(FIGURE_MAX_HEIGHT, FIGURE_BASE_HEIGHT + metric_count * FIGURE_HEIGHT_PER_METRIC),
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, figure_height), facecolor=CANVAS)
    add_round_card(fig)
    plot_left = 0.27
    plot_right = 0.92
    header_left = 0.055
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.68, bottom=0.12)
    ax.set_facecolor(SURFACE_CARD)

    top_speedup = max(row["speedup"] for row in rows)
    median_speedup = sorted(row["speedup"] for row in rows)[metric_count // 2]
    max_time = max(row["original"] for row in rows)
    x_max = max_time * 1.22
    group_gap = BAR_GROUP_GAP
    y_positions = [index * group_gap for index in range(metric_count)]
    pair_offset = BAR_PAIR_OFFSET
    bar_height = BAR_HEIGHT

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
    ax.set_yticklabels(labels, color=BODY_STRONG, fontsize=METRIC_LABEL_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_ylim(y_positions[-1] + group_gap * 0.8, -group_gap * 0.9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis="y", colors=BODY_STRONG)
    ax.xaxis.grid(True, color=HAIRLINE, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average execution time (seconds)", color=BODY, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    offset = x_max * 0.012
    optimized_label_floor = x_max * 0.035
    multiplier_x = x_max * 0.965
    leader_end_x = x_max * 0.90
    ax.text(
        multiplier_x,
        y_positions[0] - group_gap * 0.85,
        "Speedup (x)",
        va="center",
        ha="right",
        color=BODY,
        fontsize=SPEEDUP_HEADER_FONT_SIZE,
        clip_on=False,
    )
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
            fontsize=BAR_LABEL_FONT_SIZE,
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
            fontsize=BAR_LABEL_FONT_SIZE,
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
            fontsize=SPEEDUP_VALUE_FONT_SIZE,
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
        0.955,
        comparison_title(suite, backend),
        color=ON_DARK,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.900,
        comparison_subtitle(original_payload, optimized_payload, suite, backend, size_label),
        color=BODY,
        fontsize=SUBTITLE_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.872,
        f"Median {median_speedup:.2f}x across {metric_count} shared benchmarks / lower is better",
        color=MUTED,
        fontsize=DETAIL_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.846,
        comparison_context(original_payload, optimized_payload),
        color=MUTED,
        fontsize=CONTEXT_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.795,
        f"up to {top_speedup:.2f}x faster",
        color=PRIMARY,
        fontsize=CALLOUT_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(header_left, 0.735),
        ncols=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.45,
        labelspacing=0.45,
        columnspacing=1.35,
        handlelength=2.0,
        handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footer = format_env_footer(optimized_payload)
    if footer:
        fig.text(0.5, 0.048, footer, color=MUTED, fontsize=FOOTER_FONT_SIZE, ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=CANVAS, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def draw_standalone_plot(
    rows: list[dict[str, Any]],
    *,
    optimized_payload: dict[str, Any],
    suite: str,
    size_label: str,
    output_path: Path,
) -> None:
    metric_count = len(rows)
    figure_height = max(
        FIGURE_MIN_HEIGHT,
        min(FIGURE_MAX_HEIGHT, FIGURE_BASE_HEIGHT + metric_count * FIGURE_HEIGHT_PER_METRIC),
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, figure_height), facecolor=CANVAS)
    add_round_card(fig)
    plot_left = 0.27
    plot_right = 0.92
    header_left = 0.055
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.68, bottom=0.12)
    ax.set_facecolor(SURFACE_CARD)

    max_time = max(row["optimized"] for row in rows)
    x_max = max_time * 1.22
    group_gap = BAR_GROUP_GAP
    y_positions = [index * group_gap for index in range(metric_count)]
    bar_height = BAR_HEIGHT * 1.4

    optimized_values = [row["optimized"] for row in rows]
    labels = [display_metric_name(row["metric"]) for row in rows]

    ax.barh(
        y_positions,
        optimized_values,
        height=bar_height,
        color=PRIMARY,
        edgecolor=PRIMARY,
        linewidth=0.6,
        label="skmob2",
    )

    ax.set_xlim(0, x_max)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=BODY_STRONG, fontsize=METRIC_LABEL_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_ylim(y_positions[-1] + group_gap * 0.8, -group_gap * 0.9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis="y", colors=BODY_STRONG)
    ax.xaxis.grid(True, color=HAIRLINE, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average execution time (seconds)", color=BODY, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    offset = x_max * 0.012
    min_label_x = x_max * 0.035
    for group_y, row in zip(y_positions, rows, strict=True):
        label_x = row["optimized"] + offset
        if label_x < min_label_x:
            label_x = min_label_x
        ax.text(
            min(label_x, x_max * 0.985),
            group_y,
            format_seconds(row["optimized"]),
            va="center",
            ha="left",
            color=PRIMARY,
            fontsize=BAR_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    metadata = optimized_payload.get("metadata", {})
    iterations = metadata.get("iterations")
    subtitle_parts = [suite.title(), size_label]
    if iterations:
        subtitle_parts.append(f"{iterations} iterations")
    subtitle = " / ".join(subtitle_parts)

    fig.text(
        header_left,
        0.955,
        "skmob2 scaling",
        color=ON_DARK,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.900,
        subtitle,
        color=BODY,
        fontsize=SUBTITLE_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.872,
        f"{metric_count} benchmarks / lower is better",
        color=MUTED,
        fontsize=DETAIL_FONT_SIZE,
        ha="left",
        va="top",
    )
    fastest_time = min(row["optimized"] for row in rows)
    slowest_time = max(row["optimized"] for row in rows)
    fig.text(
        header_left,
        0.795,
        format_seconds(slowest_time),
        color=PRIMARY,
        fontsize=CALLOUT_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.846,
        f"range {format_seconds(fastest_time)} – {format_seconds(slowest_time)}",
        color=MUTED,
        fontsize=CONTEXT_FONT_SIZE,
        ha="left",
        va="top",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(header_left, 0.735),
        ncols=1,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.45,
        labelspacing=0.45,
        handlelength=2.0,
        handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footer = format_env_footer(optimized_payload)
    if footer:
        fig.text(0.5, 0.048, footer, color=MUTED, fontsize=FOOTER_FONT_SIZE, ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=CANVAS, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def format_seconds(value: float) -> str:
    if value < 0.01:
        return f"{value * 1_000:.2f} ms"
    if value < 1:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def format_mb(value: float) -> str:
    if value < 0.1:
        return f"{value * 1_000:.1f} KB"
    if value >= 1_000:
        return f"{value / 1_000:.2f} GB"
    return f"{value:.2f} MB"


def invalid_memory_metric_reason(metric_result: dict[str, Any], label: str) -> str | None:
    status = metric_result.get("status")
    average_mb = metric_result.get("maximum_peak_memory_mb")
    if status not in (None, "ok"):
        reason = metric_result.get("reason")
        if reason:
            return f"{label} status={status} ({reason})"
        return f"{label} status={status}"
    if not is_valid_time(average_mb):
        return f"{label} maximum_peak_memory_mb is not a positive finite number"
    return None


def missing_memory_metric_reasons(
    original_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    metric_names: list[str],
) -> list[tuple[str, str]]:
    reasons = []
    for metric in metric_names:
        orig = original_metrics.get(metric)
        opt = optimized_metrics.get(metric)
        if orig is None:
            reasons.append((metric, "missing from original skmob JSON"))
            continue
        if opt is None:
            reasons.append((metric, "missing from skmob2 JSON"))
            continue
        r1 = invalid_memory_metric_reason(orig, "original skmob")
        r2 = invalid_memory_metric_reason(opt, "skmob2")
        if r1:
            reasons.append((metric, r1))
        if r2:
            reasons.append((metric, r2))
    return reasons


def memory_metric_rows(
    original_result: dict[str, Any],
    optimized_result: dict[str, Any],
    sort_mode: str,
    expected_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    original_metrics = original_result.get("metrics", {})
    optimized_metrics = optimized_result.get("metrics", {})
    metric_names = expected_metrics or sorted(set(original_metrics).intersection(optimized_metrics))
    rows = []
    missing = missing_memory_metric_reasons(original_metrics, optimized_metrics, metric_names)
    if missing:
        reason_lines = "; ".join(f"{m}: {r}" for m, r in missing)
        raise ValueError(f"Cannot plot incomplete benchmark result: {reason_lines}")
    for metric in metric_names:
        orig_mb = float(original_metrics[metric]["maximum_peak_memory_mb"])
        opt_mb = float(optimized_metrics[metric]["maximum_peak_memory_mb"])
        reduction_pct = (orig_mb - opt_mb) / orig_mb * 100.0
        rows.append({"metric": metric, "original": orig_mb, "optimized": opt_mb, "reduction_pct": reduction_pct})
    if sort_mode == "speedup":
        rows.sort(key=lambda r: r["reduction_pct"], reverse=True)
    else:
        rows.sort(key=lambda r: r["original"], reverse=True)
    return rows


def standalone_metric_rows(
    optimized_result: dict[str, Any],
    sort_mode: str,
    expected_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    optimized_metrics = optimized_result.get("metrics", {})
    metric_names = expected_metrics or sorted(optimized_metrics)
    rows = []
    for metric in metric_names:
        metric_result = optimized_metrics.get(metric)
        if metric_result is None or invalid_metric_reason(metric_result, "skmob2"):
            continue
        rows.append({"metric": metric, "optimized": float(metric_result["average_seconds"])})
    rows.sort(key=lambda row: row["optimized"], reverse=(sort_mode != "speedup"))
    return rows


def standalone_memory_metric_rows(
    optimized_result: dict[str, Any],
    sort_mode: str,
    expected_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    optimized_metrics = optimized_result.get("metrics", {})
    metric_names = expected_metrics or sorted(optimized_metrics)
    rows = []
    for metric in metric_names:
        metric_result = optimized_metrics.get(metric)
        if metric_result is None or invalid_memory_metric_reason(metric_result, "skmob2"):
            continue
        rows.append({"metric": metric, "optimized": float(metric_result["maximum_peak_memory_mb"])})
    rows.sort(key=lambda row: row["optimized"], reverse=(sort_mode != "speedup"))
    return rows


def draw_memory_plot(
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
    figure_height = max(
        FIGURE_MIN_HEIGHT,
        min(FIGURE_MAX_HEIGHT, FIGURE_BASE_HEIGHT + metric_count * FIGURE_HEIGHT_PER_METRIC),
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, figure_height), facecolor=CANVAS)
    add_round_card(fig)
    plot_left = 0.27
    plot_right = 0.92
    header_left = 0.055
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.68, bottom=0.12)
    ax.set_facecolor(SURFACE_CARD)

    top_reduction = max(row["reduction_pct"] for row in rows)
    median_reduction = sorted(row["reduction_pct"] for row in rows)[metric_count // 2]
    max_mem = max(row["original"] for row in rows)
    x_max = max_mem * 1.22
    group_gap = BAR_GROUP_GAP
    y_positions = [index * group_gap for index in range(metric_count)]
    pair_offset = BAR_PAIR_OFFSET
    bar_height = BAR_HEIGHT

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
    ax.set_yticklabels(labels, color=BODY_STRONG, fontsize=METRIC_LABEL_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_ylim(y_positions[-1] + group_gap * 0.8, -group_gap * 0.9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis="y", colors=BODY_STRONG)
    ax.xaxis.grid(True, color=HAIRLINE, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average peak memory (MB)", color=BODY, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    offset = x_max * 0.012
    optimized_label_floor = x_max * 0.035
    multiplier_x = x_max * 0.965
    leader_end_x = x_max * 0.90
    ax.text(
        multiplier_x,
        y_positions[0] - group_gap * 0.85,
        "Reduction (%)",
        va="center",
        ha="right",
        color=BODY,
        fontsize=SPEEDUP_HEADER_FONT_SIZE,
        clip_on=False,
    )
    for index, (group_y, row) in enumerate(zip(y_positions, rows, strict=True)):
        original_y = group_y - pair_offset
        optimized_y = group_y + pair_offset
        original_label_inside = row["original"] >= x_max * 0.06
        original_label_x = row["original"] - offset if original_label_inside else row["original"] + offset
        ax.text(
            original_label_x,
            original_y,
            format_mb(row["original"]),
            va="center",
            ha="right" if original_label_inside else "left",
            color=ON_DARK,
            fontsize=BAR_LABEL_FONT_SIZE,
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
            format_mb(row["optimized"]),
            va="center",
            ha="left",
            color=PRIMARY,
            fontsize=BAR_LABEL_FONT_SIZE,
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
        reduction_pct = row["reduction_pct"]
        badge_color = PRIMARY if reduction_pct >= 0 else NEGATIVE_COLOR
        badge_text_color = CANVAS if reduction_pct >= 0 else ON_DARK
        ax.text(
            multiplier_x,
            group_y,
            f"{reduction_pct:.1f}%",
            va="center",
            ha="right",
            color=badge_text_color,
            fontsize=SPEEDUP_VALUE_FONT_SIZE,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.14",
                "facecolor": badge_color,
                "edgecolor": badge_color,
                "linewidth": 0,
            },
        )

    fig.text(
        header_left,
        0.955,
        comparison_title(suite, backend),
        color=ON_DARK,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.900,
        comparison_subtitle(original_payload, optimized_payload, suite, backend, size_label),
        color=BODY,
        fontsize=SUBTITLE_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.872,
        f"Median {median_reduction:.1f}% reduction across {metric_count} shared benchmarks / lower is better",
        color=MUTED,
        fontsize=DETAIL_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.846,
        comparison_context(original_payload, optimized_payload),
        color=MUTED,
        fontsize=CONTEXT_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.795,
        f"up to {top_reduction:.1f}% less memory",
        color=PRIMARY,
        fontsize=CALLOUT_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(header_left, 0.735),
        ncols=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.45,
        labelspacing=0.45,
        columnspacing=1.35,
        handlelength=2.0,
        handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footer = format_env_footer(optimized_payload)
    if footer:
        fig.text(0.5, 0.048, footer, color=MUTED, fontsize=FOOTER_FONT_SIZE, ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=CANVAS, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def draw_memory_standalone_plot(
    rows: list[dict[str, Any]],
    *,
    optimized_payload: dict[str, Any],
    suite: str,
    size_label: str,
    output_path: Path,
) -> None:
    metric_count = len(rows)
    figure_height = max(
        FIGURE_MIN_HEIGHT,
        min(FIGURE_MAX_HEIGHT, FIGURE_BASE_HEIGHT + metric_count * FIGURE_HEIGHT_PER_METRIC),
    )
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, figure_height), facecolor=CANVAS)
    add_round_card(fig)
    plot_left = 0.27
    plot_right = 0.92
    header_left = 0.055
    fig.subplots_adjust(left=plot_left, right=plot_right, top=0.68, bottom=0.12)
    ax.set_facecolor(SURFACE_CARD)

    max_mem = max(row["optimized"] for row in rows)
    x_max = max_mem * 1.22
    group_gap = BAR_GROUP_GAP
    y_positions = [index * group_gap for index in range(metric_count)]
    bar_height = BAR_HEIGHT * 1.4

    optimized_values = [row["optimized"] for row in rows]
    labels = [display_metric_name(row["metric"]) for row in rows]

    ax.barh(
        y_positions,
        optimized_values,
        height=bar_height,
        color=PRIMARY,
        edgecolor=PRIMARY,
        linewidth=0.6,
        label="skmob2",
    )

    ax.set_xlim(0, x_max)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=BODY_STRONG, fontsize=METRIC_LABEL_FONT_SIZE)
    ax.invert_yaxis()
    ax.set_ylim(y_positions[-1] + group_gap * 0.8, -group_gap * 0.9)
    ax.tick_params(axis="x", colors=MUTED, labelsize=AXIS_TICK_FONT_SIZE)
    ax.tick_params(axis="y", colors=BODY_STRONG)
    ax.xaxis.grid(True, color=HAIRLINE, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average peak memory (MB)", color=BODY, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    offset = x_max * 0.012
    min_label_x = x_max * 0.035
    for group_y, row in zip(y_positions, rows, strict=True):
        label_x = row["optimized"] + offset
        if label_x < min_label_x:
            label_x = min_label_x
        ax.text(
            min(label_x, x_max * 0.985),
            group_y,
            format_mb(row["optimized"]),
            va="center",
            ha="left",
            color=PRIMARY,
            fontsize=BAR_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    metadata = optimized_payload.get("metadata", {})
    iterations = metadata.get("iterations")
    subtitle_parts = [suite.title(), size_label]
    if iterations:
        subtitle_parts.append(f"{iterations} iterations")
    subtitle = " / ".join(subtitle_parts)

    fig.text(
        header_left,
        0.955,
        "skmob2 memory",
        color=ON_DARK,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.900,
        subtitle,
        color=BODY,
        fontsize=SUBTITLE_FONT_SIZE,
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.872,
        f"{metric_count} benchmarks / lower is better",
        color=MUTED,
        fontsize=DETAIL_FONT_SIZE,
        ha="left",
        va="top",
    )
    fastest_mem = min(row["optimized"] for row in rows)
    slowest_mem = max(row["optimized"] for row in rows)
    fig.text(
        header_left,
        0.795,
        format_mb(slowest_mem),
        color=PRIMARY,
        fontsize=CALLOUT_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        header_left,
        0.846,
        f"range {format_mb(fastest_mem)} – {format_mb(slowest_mem)}",
        color=MUTED,
        fontsize=CONTEXT_FONT_SIZE,
        ha="left",
        va="top",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(header_left, 0.735),
        ncols=1,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        borderpad=0.45,
        labelspacing=0.45,
        handlelength=2.0,
        handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(BODY)

    footer = format_env_footer(optimized_payload)
    if footer:
        fig.text(0.5, 0.048, footer, color=MUTED, fontsize=FOOTER_FONT_SIZE, ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=CANVAS, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _parse_large_spec_name(spec_name: str) -> tuple[str, int] | None:
    m = _LARGE_SPEC_RE.match(spec_name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _merge_location_model_data(
    standard: dict[str, Any],
    large: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in standard.get("results", []):
        if item.get("benchmark_group") == "location_models":
            n_locs = item.get("n_locations")
            if isinstance(n_locs, int):
                result[n_locs] = item
    if large:
        for item in large.get("results", []):
            if item.get("benchmark_group") == "location_models":
                n_locs = item.get("n_locations")
                if isinstance(n_locs, int):
                    result[n_locs] = item
    return result


def _merge_trajectory_model_data(
    standard: dict[str, Any],
    large: dict[str, Any] | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for item in standard.get("results", []):
        if item.get("benchmark_group") == "trajectory_models":
            n_agents = item.get("n_agents")
            n_locs = item.get("n_locations")
            if isinstance(n_agents, int) and isinstance(n_locs, int):
                result[(n_locs, n_agents)] = item.get("metrics", {})
    if large:
        for size_record in large.get("results", []):
            n_locs = size_record.get("locations") or size_record.get("size")
            if n_locs is None:
                continue
            n_locs = int(n_locs)
            for spec_name, metric_result in size_record.get("metrics", {}).items():
                parsed = _parse_large_spec_name(spec_name)
                if parsed is None:
                    continue
                model_name, n_agents = parsed
                key = (n_locs, n_agents)
                if key not in result:
                    result[key] = {}
                result[key][model_name] = metric_result
    return result


def _location_only_rows(
    orig_by_loc: dict[int, dict[str, Any]],
    opt_by_loc: dict[int, dict[str, Any]],
    n_locations: int,
) -> list[dict[str, Any]]:
    orig_metrics = orig_by_loc.get(n_locations, {}).get("metrics", {})
    opt_metrics = opt_by_loc.get(n_locations, {}).get("metrics", {})
    rows = []
    for model in MODEL_LOCATION_METRICS:
        orig_m = orig_metrics.get(model)
        opt_m = opt_metrics.get(model)
        if not orig_m or not opt_m:
            continue
        orig_t = orig_m.get("average_seconds")
        opt_t = opt_m.get("average_seconds")
        if not is_valid_time(orig_t) or not is_valid_time(opt_t):
            continue
        rows.append(
            {
                "metric": display_metric_name(model),
                "original": float(orig_t),
                "optimized": float(opt_t),
                "speedup": float(orig_t) / float(opt_t),
            }
        )
    rows.sort(key=lambda r: r["original"], reverse=True)
    return rows


def _location_only_standalone_rows(
    opt_by_loc: dict[int, dict[str, Any]],
    n_locations: int,
) -> list[dict[str, Any]]:
    opt_metrics = opt_by_loc.get(n_locations, {}).get("metrics", {})
    rows = []
    for model in MODEL_LOCATION_METRICS:
        opt_m = opt_metrics.get(model)
        if not opt_m:
            continue
        opt_t = opt_m.get("average_seconds")
        if not is_valid_time(opt_t):
            continue
        rows.append({"metric": display_metric_name(model), "optimized": float(opt_t)})
    rows.sort(key=lambda r: r["optimized"], reverse=True)
    return rows


def _agent_based_standalone_rows(
    opt_traj: dict[tuple[int, int], dict[str, Any]],
    n_agents: int,
    model_names: list[str],
) -> list[dict[str, Any]]:
    opt_locs = sorted(loc for (loc, ag) in opt_traj if ag == n_agents)
    rows = []
    for n_locs in opt_locs:
        opt_metrics = opt_traj.get((n_locs, n_agents), {})
        for model in model_names:
            opt_m = opt_metrics.get(model)
            if not opt_m:
                continue
            opt_t = opt_m.get("average_seconds")
            if not is_valid_time(opt_t):
                continue
            loc_label = format_size_label(n_locs)
            rows.append({"metric": f"{display_metric_name(model)} / {loc_label} locs", "optimized": float(opt_t)})
    rows.sort(key=lambda r: (
        parse_size_key(r["metric"].split("/")[1].strip().split(" ")[0])[1],
        -r["optimized"],
    ))
    return rows


def _agent_based_rows(
    orig_traj: dict[tuple[int, int], dict[str, Any]],
    opt_traj: dict[tuple[int, int], dict[str, Any]],
    n_agents: int,
    model_names: list[str],
) -> list[dict[str, Any]]:
    orig_locs = {loc for (loc, ag) in orig_traj if ag == n_agents}
    opt_locs = {loc for (loc, ag) in opt_traj if ag == n_agents}
    common_locs = sorted(orig_locs & opt_locs)
    rows = []
    for n_locs in common_locs:
        orig_metrics = orig_traj.get((n_locs, n_agents), {})
        opt_metrics = opt_traj.get((n_locs, n_agents), {})
        for model in model_names:
            orig_m = orig_metrics.get(model)
            opt_m = opt_metrics.get(model)
            if not orig_m or not opt_m:
                continue
            orig_t = orig_m.get("average_seconds")
            opt_t = opt_m.get("average_seconds")
            if not is_valid_time(orig_t) or not is_valid_time(opt_t):
                continue
            loc_label = format_size_label(n_locs)
            rows.append(
                {
                    "metric": f"{display_metric_name(model)} / {loc_label} locs",
                    "original": float(orig_t),
                    "optimized": float(opt_t),
                    "speedup": float(orig_t) / float(opt_t),
                }
            )
    rows.sort(key=lambda r: (
        parse_size_key(r["metric"].split("/")[1].strip().split(" ")[0])[1],
        -r["original"],
    ))
    return rows


def _memory_location_only_rows(
    orig_by_loc: dict[int, dict[str, Any]],
    opt_by_loc: dict[int, dict[str, Any]],
    n_locations: int,
) -> list[dict[str, Any]]:
    orig_metrics = orig_by_loc.get(n_locations, {}).get("metrics", {})
    opt_metrics = opt_by_loc.get(n_locations, {}).get("metrics", {})
    rows = []
    for model in MODEL_LOCATION_METRICS:
        orig_m = orig_metrics.get(model)
        opt_m = opt_metrics.get(model)
        if not orig_m or not opt_m:
            continue
        orig_mb = orig_m.get("maximum_peak_memory_mb")
        opt_mb = opt_m.get("maximum_peak_memory_mb")
        if not is_valid_time(orig_mb) or not is_valid_time(opt_mb):
            continue
        reduction_pct = (float(orig_mb) - float(opt_mb)) / float(orig_mb) * 100.0
        rows.append(
            {
                "metric": display_metric_name(model),
                "original": float(orig_mb),
                "optimized": float(opt_mb),
                "reduction_pct": reduction_pct,
            }
        )
    rows.sort(key=lambda r: r["original"], reverse=True)
    return rows


def _memory_location_only_standalone_rows(
    opt_by_loc: dict[int, dict[str, Any]],
    n_locations: int,
) -> list[dict[str, Any]]:
    opt_metrics = opt_by_loc.get(n_locations, {}).get("metrics", {})
    rows = []
    for model in MODEL_LOCATION_METRICS:
        opt_m = opt_metrics.get(model)
        if not opt_m:
            continue
        opt_mb = opt_m.get("maximum_peak_memory_mb")
        if not is_valid_time(opt_mb):
            continue
        rows.append({"metric": display_metric_name(model), "optimized": float(opt_mb)})
    rows.sort(key=lambda r: r["optimized"], reverse=True)
    return rows


def _memory_agent_based_standalone_rows(
    opt_traj: dict[tuple[int, int], dict[str, Any]],
    n_agents: int,
    model_names: list[str],
) -> list[dict[str, Any]]:
    opt_locs = sorted(loc for (loc, ag) in opt_traj if ag == n_agents)
    rows = []
    for n_locs in opt_locs:
        opt_metrics = opt_traj.get((n_locs, n_agents), {})
        for model in model_names:
            opt_m = opt_metrics.get(model)
            if not opt_m:
                continue
            opt_mb = opt_m.get("maximum_peak_memory_mb")
            if not is_valid_time(opt_mb):
                continue
            loc_label = format_size_label(n_locs)
            rows.append({"metric": f"{display_metric_name(model)} / {loc_label} locs", "optimized": float(opt_mb)})
    rows.sort(key=lambda r: (
        parse_size_key(r["metric"].split("/")[1].strip().split(" ")[0])[1],
        -r["optimized"],
    ))
    return rows


def _memory_agent_based_rows(
    orig_traj: dict[tuple[int, int], dict[str, Any]],
    opt_traj: dict[tuple[int, int], dict[str, Any]],
    n_agents: int,
    model_names: list[str],
) -> list[dict[str, Any]]:
    orig_locs = {loc for (loc, ag) in orig_traj if ag == n_agents}
    opt_locs = {loc for (loc, ag) in opt_traj if ag == n_agents}
    common_locs = sorted(orig_locs & opt_locs)
    rows = []
    for n_locs in common_locs:
        orig_metrics = orig_traj.get((n_locs, n_agents), {})
        opt_metrics = opt_traj.get((n_locs, n_agents), {})
        for model in model_names:
            orig_m = orig_metrics.get(model)
            opt_m = opt_metrics.get(model)
            if not orig_m or not opt_m:
                continue
            orig_mb = orig_m.get("maximum_peak_memory_mb")
            opt_mb = opt_m.get("maximum_peak_memory_mb")
            if not is_valid_time(orig_mb) or not is_valid_time(opt_mb):
                continue
            reduction_pct = (float(orig_mb) - float(opt_mb)) / float(orig_mb) * 100.0
            loc_label = format_size_label(n_locs)
            rows.append(
                {
                    "metric": f"{display_metric_name(model)} / {loc_label} locs",
                    "original": float(orig_mb),
                    "optimized": float(opt_mb),
                    "reduction_pct": reduction_pct,
                }
            )
    rows.sort(key=lambda r: (
        parse_size_key(r["metric"].split("/")[1].strip().split(" ")[0])[1],
        -r["original"],
    ))
    return rows


def generate_model_memory_plots(args: argparse.Namespace) -> int:
    optimized_path = args.optimized_json.resolve()
    if not optimized_path.exists():
        print(f"ERROR: optimized benchmark JSON not found: {optimized_path}")
        return 2

    optimized_payload = load_json(optimized_path)
    if args.standalone:
        original_payload = {"metadata": {}, "results": []}
    else:
        if args.original_json is None:
            print("ERROR: --original-json is required unless --standalone is used.")
            return 2
        original_path = args.original_json.resolve()
        if not original_path.exists():
            print(f"ERROR: original benchmark JSON not found: {original_path}")
            return 2
        original_payload = load_json(original_path)

    def _maybe_load(attr: str) -> dict[str, Any] | None:
        p: Path | None = getattr(args, attr, None)
        if p is None:
            return None
        rp = p.resolve()
        if not rp.exists():
            return None
        return load_json(rp)

    orig_large = None if args.standalone else _maybe_load("large_json_skmob")
    opt_large = _maybe_load("large_json_skmob2")
    orig_loc_large = None if args.standalone else _maybe_load("large_loc_json_skmob")
    opt_loc_large = _maybe_load("large_loc_json_skmob2")

    orig_loc = _merge_location_model_data(original_payload, orig_loc_large)
    opt_loc = _merge_location_model_data(optimized_payload, opt_loc_large)
    orig_traj = _merge_trajectory_model_data(original_payload, orig_large)
    opt_traj = _merge_trajectory_model_data(optimized_payload, opt_large)

    generated = 0

    def _loc_safe(n: int) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", format_size_label(n))

    common_loc_counts = sorted(set(orig_loc) & set(opt_loc))
    for n_locs in common_loc_counts:
        rows = _memory_location_only_rows(orig_loc, opt_loc, n_locs)
        if not rows:
            continue
        loc_label = format_size_label(n_locs)
        size_label = f"Location-only / {loc_label} locations"
        out_name = f"model_location_only_{_loc_safe(n_locs)}_locations_memory.png"
        output_path = args.output_dir / out_name
        draw_memory_plot(
            rows,
            original_payload=original_payload,
            optimized_payload=optimized_payload,
            suite="models",
            backend=None,
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    skmob2_only_loc_counts = sorted(set(opt_loc) - set(orig_loc))
    for n_locs in skmob2_only_loc_counts:
        standalone_rows = _memory_location_only_standalone_rows(opt_loc, n_locs)
        if not standalone_rows:
            continue
        loc_label = format_size_label(n_locs)
        size_label = f"Location-only / {loc_label} locations"
        out_name = f"model_location_only_{_loc_safe(n_locs)}_locations_skmob2_memory.png"
        output_path = args.output_dir / out_name
        draw_memory_standalone_plot(
            standalone_rows,
            optimized_payload=optimized_payload,
            suite="models",
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    all_agent_counts = sorted(
        {ag for (_, ag) in orig_traj} & {ag for (_, ag) in opt_traj}
    )
    for n_agents in all_agent_counts:
        rows = _memory_agent_based_rows(orig_traj, opt_traj, n_agents, MODEL_TRAJECTORY_METRICS)
        if not rows:
            continue
        agent_label = format_size_label(n_agents)
        size_label = f"Agent-based / {agent_label} agents"
        out_name = f"model_agent_based_{re.sub(r'[^A-Za-z0-9_.-]+', '_', agent_label)}_agents_memory.png"
        output_path = args.output_dir / out_name
        draw_memory_plot(
            rows,
            original_payload=original_payload,
            optimized_payload=optimized_payload,
            suite="models",
            backend=None,
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    skmob2_only_agents = sorted(
        {ag for (_, ag) in opt_traj} - {ag for (_, ag) in orig_traj}
    )
    for n_agents in skmob2_only_agents:
        standalone_rows = _memory_agent_based_standalone_rows(opt_traj, n_agents, MODEL_TRAJECTORY_METRICS)
        if not standalone_rows:
            continue
        agent_label = format_size_label(n_agents)
        size_label = f"Agent-based / {agent_label} agents"
        out_name = f"model_agent_based_{re.sub(r'[^A-Za-z0-9_.-]+', '_', agent_label)}_agents_skmob2_memory.png"
        output_path = args.output_dir / out_name
        draw_memory_standalone_plot(
            standalone_rows,
            optimized_payload=optimized_payload,
            suite="models",
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    if generated == 0:
        print("No model memory plots were generated.")
        return 1
    return 0


def generate_memory_plots(args: argparse.Namespace) -> int:
    if args.standalone:
        optimized_path = args.optimized_json.resolve()
        if not optimized_path.exists():
            print(f"ERROR: optimized benchmark JSON not found: {optimized_path}")
            return 2
        optimized_payload = load_json(optimized_path)
        if args.suite == "models":
            return generate_model_memory_plots(args)

        optimized_results = result_map(optimized_payload)
        labels = sorted(optimized_results, key=parse_size_key)
        if args.sizes:
            requested = [format_size_label(size) for size in args.sizes]
            labels = [label for label in requested if label in optimized_results]
        expected_metrics = expected_metrics_from_catalog(args.catalog, args.suite)
        out_dir = _resolve_output_dir(args.output_dir, optimized_payload)
        generated = 0
        for label in labels:
            rows = standalone_memory_metric_rows(optimized_results[label], args.sort, expected_metrics)
            if not rows:
                print(f"No valid skmob2 memory metrics found for {label}; skipping.")
                continue
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            backend_part = f"_{args.backend}" if args.backend else ""
            output_path = out_dir / f"skmob2_{args.suite}{backend_part}_{safe_label}_memory.png"
            draw_memory_standalone_plot(
                rows,
                optimized_payload=optimized_payload,
                suite=args.suite,
                size_label=label,
                output_path=output_path,
            )
            generated += 1
            print(f"Saved {output_path}")
        if generated == 0:
            print("No standalone memory plots were generated.")
            return 1
        return 0

    if args.original_json is None:
        print("ERROR: --original-json is required unless --standalone is used.")
        return 2
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
    if args.suite == "models":
        return generate_model_memory_plots(args)

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

    out_dir = _resolve_output_dir(args.output_dir, original_payload, optimized_payload)
    generated = 0
    for label in labels:
        try:
            rows = memory_metric_rows(original_results[label], optimized_results[label], args.sort, expected_metrics)
        except ValueError as exc:
            print(f"ERROR for {label}: {exc}")
            return 1
        if not rows:
            print(f"No valid overlapping metrics found for {label}; skipping.")
            continue
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        if not args.backend:
            out_name = f"skmob2_vs_skmob_{args.suite}_{safe_label}_memory.png"
        else:
            out_name = f"skmob2_vs_skmob_{args.suite}_{args.backend}_{safe_label}_memory.png"
        output_path = out_dir / out_name
        draw_memory_plot(
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
        print("No memory plots were generated.")
        return 1
    return 0


def generate_model_plots(args: argparse.Namespace) -> int:
    optimized_path = args.optimized_json.resolve()
    if not optimized_path.exists():
        print(f"ERROR: optimized benchmark JSON not found: {optimized_path}")
        return 2

    optimized_payload = load_json(optimized_path)
    if args.standalone:
        original_payload = {"metadata": {}, "results": []}
    else:
        if args.original_json is None:
            print("ERROR: --original-json is required unless --standalone is used.")
            return 2
        original_path = args.original_json.resolve()
        if not original_path.exists():
            print(f"ERROR: original benchmark JSON not found: {original_path}")
            return 2
        original_payload = load_json(original_path)

    def _maybe_load(attr: str) -> dict[str, Any] | None:
        p: Path | None = getattr(args, attr, None)
        if p is None:
            return None
        rp = p.resolve()
        if not rp.exists():
            return None
        return load_json(rp)

    orig_large = None if args.standalone else _maybe_load("large_json_skmob")
    opt_large = _maybe_load("large_json_skmob2")
    orig_loc_large = None if args.standalone else _maybe_load("large_loc_json_skmob")
    opt_loc_large = _maybe_load("large_loc_json_skmob2")

    orig_loc = _merge_location_model_data(original_payload, orig_loc_large)
    opt_loc = _merge_location_model_data(optimized_payload, opt_loc_large)
    orig_traj = _merge_trajectory_model_data(original_payload, orig_large)
    opt_traj = _merge_trajectory_model_data(optimized_payload, opt_large)

    generated = 0

    def _loc_safe(n: int) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", format_size_label(n))

    # Family 1 — Location-only models: one chart per location count
    # Comparison charts where both libraries have data
    common_loc_counts = sorted(set(orig_loc) & set(opt_loc))
    for n_locs in common_loc_counts:
        rows = _location_only_rows(orig_loc, opt_loc, n_locs)
        if not rows:
            continue
        loc_label = format_size_label(n_locs)
        size_label = f"Location-only / {loc_label} locations"
        out_name = f"model_location_only_{_loc_safe(n_locs)}_locations.png"
        output_path = args.output_dir / out_name
        draw_plot(
            rows,
            original_payload=original_payload,
            optimized_payload=optimized_payload,
            suite="models",
            backend=None,
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    # Standalone skmob2-only charts for location counts not available in skmob
    skmob2_only_loc_counts = sorted(set(opt_loc) - set(orig_loc))
    for n_locs in skmob2_only_loc_counts:
        standalone_rows = _location_only_standalone_rows(opt_loc, n_locs)
        if not standalone_rows:
            continue
        loc_label = format_size_label(n_locs)
        size_label = f"Location-only / {loc_label} locations"
        out_name = f"model_location_only_{_loc_safe(n_locs)}_locations_skmob2.png"
        output_path = args.output_dir / out_name
        draw_standalone_plot(
            standalone_rows,
            optimized_payload=optimized_payload,
            suite="models",
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    # Family 2 — Agent-based models: one chart per agent count
    # Comparison charts where both libraries have data
    all_agent_counts = sorted(
        {ag for (_, ag) in orig_traj} & {ag for (_, ag) in opt_traj}
    )
    for n_agents in all_agent_counts:
        rows = _agent_based_rows(orig_traj, opt_traj, n_agents, MODEL_TRAJECTORY_METRICS)
        if not rows:
            continue
        agent_label = format_size_label(n_agents)
        size_label = f"Agent-based / {agent_label} agents"
        out_name = f"model_agent_based_{re.sub(r'[^A-Za-z0-9_.-]+', '_', agent_label)}_agents.png"
        output_path = args.output_dir / out_name
        draw_plot(
            rows,
            original_payload=original_payload,
            optimized_payload=optimized_payload,
            suite="models",
            backend=None,
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    # Standalone skmob2-only charts for agent counts not available in skmob
    skmob2_only_agents = sorted(
        {ag for (_, ag) in opt_traj} - {ag for (_, ag) in orig_traj}
    )
    for n_agents in skmob2_only_agents:
        standalone_rows = _agent_based_standalone_rows(opt_traj, n_agents, MODEL_TRAJECTORY_METRICS)
        if not standalone_rows:
            continue
        agent_label = format_size_label(n_agents)
        size_label = f"Agent-based / {agent_label} agents"
        out_name = f"model_agent_based_{re.sub(r'[^A-Za-z0-9_.-]+', '_', agent_label)}_agents_skmob2.png"
        output_path = args.output_dir / out_name
        draw_standalone_plot(
            standalone_rows,
            optimized_payload=optimized_payload,
            suite="models",
            size_label=size_label,
            output_path=output_path,
        )
        generated += 1
        print(f"Saved {output_path}")

    if generated == 0:
        print("No model plots were generated.")
        return 1
    return 0


def generate_plots(args: argparse.Namespace) -> int:
    if getattr(args, "profile", "speed") == "memory":
        return generate_memory_plots(args)

    if args.standalone:
        optimized_path = args.optimized_json.resolve()
        if not optimized_path.exists():
            print(f"ERROR: optimized benchmark JSON not found: {optimized_path}")
            return 2
        optimized_payload = load_json(optimized_path)
        if args.suite == "models":
            return generate_model_plots(args)

        optimized_results = result_map(optimized_payload)
        labels = sorted(optimized_results, key=parse_size_key)
        if args.sizes:
            requested = [format_size_label(size) for size in args.sizes]
            labels = [label for label in requested if label in optimized_results]
        expected_metrics = expected_metrics_from_catalog(args.catalog, args.suite)
        out_dir = _resolve_output_dir(args.output_dir, optimized_payload)
        generated = 0
        for label in labels:
            rows = standalone_metric_rows(optimized_results[label], args.sort, expected_metrics)
            if not rows:
                print(f"No valid skmob2 speed metrics found for {label}; skipping.")
                continue
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            backend_part = f"_{args.backend}" if args.backend else ""
            output_path = out_dir / f"skmob2_{args.suite}{backend_part}_{safe_label}.png"
            draw_standalone_plot(
                rows,
                optimized_payload=optimized_payload,
                suite=args.suite,
                size_label=label,
                output_path=output_path,
            )
            generated += 1
            print(f"Saved {output_path}")
        if generated == 0:
            print("No standalone plots were generated.")
            return 1
        return 0

    if args.original_json is None:
        print("ERROR: --original-json is required unless --standalone is used.")
        return 2
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
    if args.suite == "models":
        return generate_model_plots(args)

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

    out_dir = _resolve_output_dir(args.output_dir, original_payload, optimized_payload)
    generated = 0
    for label in labels:
        try:
            rows = metric_rows(
                original_results[label],
                optimized_results[label],
                args.sort,
                expected_metrics,
                skip_invalid=getattr(args, "skip_invalid", False),
            )
        except ValueError as exc:
            print(f"ERROR for {label}: {exc}")
            return 1
        if not rows:
            print(f"No valid overlapping metrics found for {label}; skipping.")
            continue
        output_path = out_dir / output_name(args.suite, args.backend, label)
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
    parser.add_argument("--original-json", type=Path, help="Original skmob benchmark JSON.")
    parser.add_argument("--optimized-json", required=True, type=Path, help="skmob2 benchmark JSON.")
    parser.add_argument("--suite", required=True, help="Benchmark suite name, for example individual or privacy.")
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
        default=Path("benchmarks/results/plots"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument("--sizes", nargs="*", help="Optional size labels such as 1M 4M.")
    parser.add_argument(
        "--sort",
        choices=("speedup", "original-time"),
        default="speedup",
        help="Metric ordering for the plot.",
    )
    parser.add_argument(
        "--large-json-skmob2",
        type=Path,
        default=None,
        dest="large_json_skmob2",
        help="Large-scale trajectory benchmark JSON for skmob2 (models suite only).",
    )
    parser.add_argument(
        "--large-json-skmob",
        type=Path,
        default=None,
        dest="large_json_skmob",
        help="Large-scale trajectory benchmark JSON for original skmob (models suite only).",
    )
    parser.add_argument(
        "--large-loc-json-skmob2",
        type=Path,
        default=None,
        dest="large_loc_json_skmob2",
        help="Large-scale location-only benchmark JSON for skmob2 (models suite only).",
    )
    parser.add_argument(
        "--large-loc-json-skmob",
        type=Path,
        default=None,
        dest="large_loc_json_skmob",
        help="Large-scale location-only benchmark JSON for original skmob (models suite only).",
    )
    parser.add_argument(
        "--profile",
        choices=("speed", "memory"),
        default="speed",
        help="Benchmark profile: speed (default) or memory.",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Generate skmob2-only plots without requiring an original skmob JSON.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip missing, skipped, or errored metrics instead of failing the whole comparison plot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    return generate_plots(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
