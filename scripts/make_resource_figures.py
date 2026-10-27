"""Resources-page figure: the two datasets used throughout the docs, drawn from the cached files.

Usage: uv run --no-project --python .venv/bin/python --with matplotlib python scripts/make_resource_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("docs/src/assets/images/resources")
DATA = Path("tests/shared/data")
GOLD, GREY, TEXT = "#c9a400", "#8a8a8a", "#777777"


def brightkite(ax):
    df = pd.read_csv(
        DATA / "loc-brightkite_totalCheckins.txt.gz",
        sep="\t",
        header=None,
        names=["uid", "datetime", "lat", "lng", "loc"],
        usecols=["lat", "lng"],
    ).dropna()
    ax.scatter(df.lng, df.lat, s=0.4, color=GOLD, alpha=0.25, linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_title("Brightkite: 4.7M check-ins worldwide", fontsize=11, color=TEXT, loc="left")
    return len(df)


def geolife(ax):
    frames = [
        pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1], names=["lat", "lng"])
        for f in sorted((DATA / "Geolife Trajectories 1.3" / "Data").glob("*/Trajectory/*.plt"))[::6]
    ]
    df = pd.concat(frames)
    df = df[df.lat.between(39.75, 40.15) & df.lng.between(116.1, 116.7)]
    ax.scatter(df.lng, df.lat, s=0.5, color=GREY, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlim(116.1, 116.7)
    ax.set_ylim(39.75, 40.15)
    ax.set_title("GeoLife: GPS traces around Beijing", fontsize=11, color=TEXT, loc="left")
    return len(df)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=170, gridspec_kw={"width_ratios": [1.7, 1]})
print(brightkite(axes[0]), geolife(axes[1]))
for ax in axes:
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.tight_layout(pad=0.4)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "resources-datasets.png", transparent=True)
