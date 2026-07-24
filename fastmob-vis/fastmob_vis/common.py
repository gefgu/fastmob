from __future__ import annotations

from typing import Any, Sequence

from .brand import FONT_MONO, FONT_SANS, FONT_SERIF, PALETTES

COLOR_ROLES = ["observed", "synthetic", "syn2", "syn3", "baseline"]


def norm_width(width: int | str) -> str:
    return f"{width}px" if isinstance(width, int) else width


def resolve_palette(palette: str | dict) -> dict:
    if isinstance(palette, dict):
        return palette
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; choices: {list(PALETTES)}")
    return PALETTES[palette]


def base_option(title: str, p: dict, chart_type: str) -> dict[str, Any]:
    return {
        "animation": False,
        "backgroundColor": p["bg"],
        "textStyle": {"fontFamily": FONT_SANS, "color": p["axis"]},
        "_meta": {"chartType": chart_type},
        "title": {
            "text": title,
            "left": 0,
            "top": 0,
            "textStyle": {
                "fontFamily": FONT_SERIF,
                "fontWeight": 500,
                "fontSize": 32,
                "color": p["axis"],
            },
        },
    }


def activity_color(p: dict, index: int) -> str:
    return p[COLOR_ROLES[index % len(COLOR_ROLES)]]


def heatmap_option(
    *,
    title: str,
    p: dict,
    chart_type: str,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    data: list[list[float]],
    vmax: float,
    x_name: str,
    y_name: str,
    tooltip_value_suffix: str,
    grid: dict[str, Any],
) -> dict[str, Any]:
    option = base_option(title, p, chart_type)
    option.update(
        {
            "grid": grid,
            "tooltip": {
                "position": "top",
                "backgroundColor": p["bg"],
                "borderColor": p["axis"],
                "borderWidth": 1.25,
                "textStyle": {"color": p["axis"], "fontFamily": FONT_SANS},
            },
            "xAxis": {
                "type": "category",
                "data": list(x_labels),
                "name": x_name,
                "nameLocation": "middle",
                "nameGap": 48,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {"fontFamily": FONT_MONO, "fontSize": 12, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
                "splitArea": {"show": False},
            },
            "yAxis": {
                "type": "category",
                "data": list(y_labels),
                "name": y_name,
                "nameLocation": "middle",
                "nameGap": 62,
                "nameRotate": 90,
                "nameTextStyle": {"fontFamily": FONT_MONO, "fontSize": 14, "color": p["axis"]},
                "axisLabel": {"fontFamily": FONT_SANS, "fontSize": 13, "color": p["axis"]},
                "axisLine": {"lineStyle": {"color": p["axis"], "width": 1.5}},
                "axisTick": {"show": False},
                "splitArea": {"show": False},
            },
            "visualMap": {
                "min": 0,
                "max": vmax,
                "calculable": True,
                "orient": "vertical",
                "right": 0,
                "top": "middle",
                "textStyle": {"fontFamily": FONT_MONO, "color": p["axis"]},
                "inRange": {"color": [p["bg"], p["band"], p["synthetic"]]},
            },
            "series": [
                {
                    "name": "Percentage",
                    "type": "heatmap",
                    "data": data,
                    "label": {
                        "show": chart_type == "transition",
                        "formatter": "{@[2]}",
                        "fontFamily": FONT_MONO,
                        "fontSize": 12,
                        "color": p["axis"],
                    },
                    "emphasis": {"itemStyle": {"borderColor": p["axis"], "borderWidth": 1}},
                }
            ],
        }
    )
    option["tooltip"]["formatter"] = "{b}<br/>Percentage: {@[2]}" + tooltip_value_suffix
    return option
