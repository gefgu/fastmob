from __future__ import annotations

import base64
import importlib.resources
from functools import lru_cache
from typing import Any, Sequence

from .brand import FONT_MONO, FONT_SANS
from .common import base_option, norm_width, resolve_palette
from .data import column, columns, data_len
from .figure import EChartsFigure

OTHER_MOTIF_ID = "other"

LITERATURE_MOTIF_PERCENTAGES: dict[int, float] = {
    1: 10.10,
    2: 30.80,
    3: 12.70,
    4: 9.40,
    5: 0.60,
    6: 7.30,
    7: 5.00,
    8: 1.70,
    9: 0.70,
    10: 3.00,
    11: 2.30,
    12: 1.30,
    13: 1.00,
    14: 1.30,
    15: 1.30,
    16: 0.72,
    17: 0.86,
}

LITERATURE_TO_FASTMOB_MOTIF_ID: dict[int, int] = {
    1: 0x1000000000,
    2: 0x2000000006,
    3: 0x30000000E4,
    4: 0x300000008C,
    5: 0x30000000E2,
    6: 0x4000006818,
    7: 0x4000004218,
    8: 0x4000007888,
    9: 0x4000006984,
    10: 0x5000C80830,
    11: 0x5000820830,
    12: 0x5000E84030,
    13: 0x5000C10610,
    14: 0x6620102060,
    15: 0x6408102060,
    16: 0x6720802060,
    17: 0x66040A0060,
}

LITERATURE_MOTIF_IDS = tuple(LITERATURE_TO_FASTMOB_MOTIF_ID.values())
LITERATURE_MOTIF_ID_TO_ORDINAL = {
    fastmob_motif_id: literature_motif_id
    for literature_motif_id, fastmob_motif_id in LITERATURE_TO_FASTMOB_MOTIF_ID.items()
}


@lru_cache(maxsize=None)
def _motif_svg_data_uri(motif_id: int) -> str:
    filename = f"{format_motif_hex_id(motif_id)}.svg"
    with importlib.resources.open_binary("fastmob_vis.assets.motifs", filename) as resource:
        encoded = base64.b64encode(resource.read()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _motif_axis_label_styles() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    label_keys = {}
    rich_styles = {}
    for ordinal, motif_id in LITERATURE_TO_FASTMOB_MOTIF_ID.items():
        hex_id = format_motif_hex_id(motif_id)
        style_key = f"motif_{ordinal}"
        label_keys[hex_id] = style_key
        rich_styles[style_key] = {
            "width": 88,
            "height": 88,
            "backgroundColor": {"image": _motif_svg_data_uri(motif_id)},
        }
    return label_keys, rich_styles


def format_motif_hex_id(motif_id: int | str | None) -> str:
    """Return a compact hex label for a packed fastmob motif ID."""
    if motif_id is None or motif_id == OTHER_MOTIF_ID:
        return OTHER_MOTIF_ID
    try:
        motif_int = int(motif_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"motif_id must be integer-like or 'other', got {motif_id!r}") from exc
    return hex(motif_int)


def _empty_basis() -> list[dict[str, Any]]:
    rows = []
    for literature_motif_id, fastmob_motif_id in LITERATURE_TO_FASTMOB_MOTIF_ID.items():
        rows.append(
            {
                "literature_motif_id": literature_motif_id,
                "motif_id": fastmob_motif_id,
                "hex_id": format_motif_hex_id(fastmob_motif_id),
                "percentage": 0.0,
                "count": 0,
            }
        )
    rows.append(
        {
            "literature_motif_id": OTHER_MOTIF_ID,
            "motif_id": OTHER_MOTIF_ID,
            "hex_id": OTHER_MOTIF_ID.title(),
            "percentage": 0.0,
            "count": 0,
        }
    )
    return rows


def _literature_distribution_rows(literature_df: Any | None) -> list[dict[str, Any]]:
    if literature_df is None:
        rows = _empty_basis()
        total = sum(LITERATURE_MOTIF_PERCENTAGES.values())
        for row in rows[:-1]:
            percentage = LITERATURE_MOTIF_PERCENTAGES[int(row["literature_motif_id"])]
            row["percentage"] = percentage
        rows[-1]["percentage"] = max(0.0, 100.0 - total)
        return rows
    if "fastmob_motif_id" in columns(literature_df):
        data: dict[str, list[Any]] = {"motif_id": column(literature_df, "fastmob_motif_id")}
        data_columns = columns(literature_df)
        if "percentage" in data_columns:
            data["percentage"] = column(literature_df, "percentage")
        if "count" in data_columns:
            data["count"] = column(literature_df, "count")
        return map_motif_distribution_to_literature_basis(data)
    return map_motif_distribution_to_literature_basis(literature_df)


def map_motif_distribution_to_literature_basis(distribution_df: Any) -> list[dict[str, Any]]:
    """Map a motif distribution to the 17 literature packed IDs plus Other."""
    data_columns = columns(distribution_df)
    if "motif_id" not in data_columns:
        raise ValueError("motif distribution must contain a 'motif_id' column")
    if "percentage" not in data_columns and "count" not in data_columns:
        raise ValueError("motif distribution must contain either 'percentage' or 'count'")

    rows = _empty_basis()
    by_motif_id = {row["motif_id"]: row for row in rows[:-1]}
    other = rows[-1]

    motif_ids = column(distribution_df, "motif_id")
    counts = column(distribution_df, "count") if "count" in data_columns else [0] * data_len(distribution_df)
    if "percentage" in data_columns:
        percentages = [float(value) for value in column(distribution_df, "percentage")]
    else:
        total_count = sum(float(value) for value in counts)
        percentages = [
            (float(value) / total_count) * 100.0 if total_count > 0 else 0.0
            for value in counts
        ]

    for motif_id, percentage, count in zip(motif_ids, percentages, counts):
        try:
            motif_key = int(motif_id)
        except (TypeError, ValueError):
            motif_key = None
        target = by_motif_id.get(motif_key, other)
        target["percentage"] = round(float(target["percentage"]) + float(percentage), 2)
        target["count"] = int(target["count"]) + int(count or 0)

    return rows


def _series_data(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row["hex_id"],
            round(float(row["percentage"]), 2),
            row["literature_motif_id"],
            row["motif_id"],
            row["hex_id"],
            row["count"],
        ]
        for row in rows
    ]


def _bar_series(name: str, rows: list[dict[str, Any]], color: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "bar",
        "barMaxWidth": 32,
        "data": _series_data(rows),
        "dimensions": [
            "hex_id",
            "percentage",
            "literature_motif_id",
            "packed_motif_id",
            "hex_label",
            "count",
        ],
        "encode": {"x": "hex_id", "y": "percentage"},
        "itemStyle": {"color": color},
        "emphasis": {"itemStyle": {"borderColor": "#000", "borderWidth": 1}},
    }


def plot_motif_literature_comparison(
    *,
    literature_df: Any | None = None,
    reference_distribution: Any | None = None,
    comparison_distribution: Any | None = None,
    labels: Sequence[str] = ("Reference", "Simulation"),
    title: str = "Motif Literature Comparison",
    palette: str | dict = "warm",
    width: int | str = 900,
    height: str = "560px",
    bundle_libs: bool = True,
    display: str = "html",
) -> EChartsFigure:
    """Build a grouped bar chart comparing motif distributions to literature."""
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two strings")
    p = resolve_palette(palette)

    literature_rows = _literature_distribution_rows(literature_df)
    categories = [row["hex_id"] for row in literature_rows]
    motif_label_keys, motif_label_styles = _motif_axis_label_styles()
    series = [_bar_series("Literature", literature_rows, p["baseline"])]
    if reference_distribution is not None:
        series.append(
            _bar_series(
                str(labels[0]),
                map_motif_distribution_to_literature_basis(reference_distribution),
                p["observed"],
            )
        )
    if comparison_distribution is not None:
        series.append(
            _bar_series(
                str(labels[1]),
                map_motif_distribution_to_literature_basis(comparison_distribution),
                p["synthetic"],
            )
        )

    option = base_option(title, p, "motif_literature_comparison")
    option["_meta"]["motifLabelKeys"] = motif_label_keys
    option.update(
        {
            "legend": {
                "top": 44,
                "right": 28,
                "textStyle": {"fontFamily": FONT_SANS, "fontSize": 13, "color": p["axis"]},
            },
            "grid": {"left": 78, "right": 28, "top": 96, "bottom": 156, "containLabel": False},
            "tooltip": {
                "trigger": "item",
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
                "formatter": (
                    "Literature motif: {@literature_motif_id}<br/>"
                    "Packed motif ID: {@packed_motif_id}<br/>"
                    "Hex ID: {@hex_label}<br/>"
                    "{a}: {@percentage}%<br/>"
                    "Count: {@count}"
                ),
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "name": "MOTIF ID",
                "nameLocation": "middle",
                "nameGap": 124,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {
                    "fontFamily": FONT_MONO,
                    "fontSize": 22,
                    "color": p["axis"],
                    "interval": 0,
                    "margin": 10,
                    "rich": motif_label_styles,
                },
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "name": "% OF USER-DAYS",
                "nameLocation": "middle",
                "nameGap": 52,
                "nameRotate": 90,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {"formatter": "{value}", "fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "splitLine": {"lineStyle": {"color": p["grid"], "type": [2, 5], "width": 1}},
            },
            "series": series,
        }
    )
    return EChartsFigure(
        option,
        background=p["bg"],
        width=norm_width(width),
        height=height,
        bundle_libs=bundle_libs,
        display=display,
    )
