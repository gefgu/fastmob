from __future__ import annotations

PALETTES: dict[str, dict[str, str]] = {
    "warm": {
        "bg":        "#fbf8f1",
        "grid":      "#dcd5c4",
        "axis":      "#14110d",
        "observed":  "#14110d",
        "synthetic": "#c8533a",
        "syn2":      "#7a8c5a",
        "syn3":      "#3d6a8a",
        "baseline":  "#a39c8e",
        "band":      "rgba(200,83,58,0.16)",
    },
    "ink": {
        "bg":        "#fbf8f1",
        "grid":      "#d8d2c2",
        "axis":      "#14110d",
        "observed":  "#14110d",
        "synthetic": "#5a544a",
        "syn2":      "#7a7468",
        "syn3":      "#403b34",
        "baseline":  "#bfb8a8",
        "band":      "rgba(20,17,13,0.10)",
    },
    "terracotta": {
        "bg":        "#fbf8f1",
        "grid":      "#e6c8be",
        "axis":      "#3a1e15",
        "observed":  "#3a1e15",
        "synthetic": "#c8533a",
        "syn2":      "#e08a4a",
        "syn3":      "#7a3325",
        "baseline":  "#b89a8e",
        "band":      "rgba(200,83,58,0.18)",
    },
    "forest": {
        "bg":        "#f4efe3",
        "grid":      "#cdc5b0",
        "axis":      "#1e2a1a",
        "observed":  "#1e2a1a",
        "synthetic": "#4e7a3d",
        "syn2":      "#8a9a4a",
        "syn3":      "#2d4a3e",
        "baseline":  "#a39c8e",
        "band":      "rgba(78,122,61,0.18)",
    },
}

FONT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400"
    "&family=IBM+Plex+Sans:wght@400;500"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&display=swap"
)

FONT_SERIF = '"IBM Plex Serif", Georgia, serif'
FONT_SANS  = '"IBM Plex Sans", system-ui, sans-serif'
FONT_MONO  = '"IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace'
