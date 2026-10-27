"""Home-page figure: one GeoLife user-day with detected staypoints and triplegs.

Needs the cached GeoLife data in tests/shared/data.
Usage: uv run --no-project --python .venv/bin/python --with matplotlib python scripts/make_home_geolife_figure.py
"""

import shutil
import tempfile
from pathlib import Path

import fastmob
import matplotlib.pyplot as plt
import pandas as pd
from fastmob import Positionfixes

OUT = Path("docs/src/assets/images/home")
DATA = Path("tests/shared/data/Geolife Trajectories 1.3/Data")
GOLD, GREY, TEXT = "#c9a400", "#8a8a8a", "#777777"

tmp = Path(tempfile.mkdtemp()) / "Geolife Trajectories 1.3" / "Data"
tmp.mkdir(parents=True)
shutil.copytree(DATA / "000", tmp / "000")
tdf = fastmob.io.load_geolife_trajectories(str(tmp.parent)).to_pandas()
tdf["datetime"] = pd.to_datetime(tdf["datetime"], utc=True)
tdf = tdf.sort_values("datetime")
tdf = tdf[tdf.lat.between(39.9, 40.1) & tdf.lng.between(116.2, 116.5)]

fixes = Positionfixes(tdf)
stays = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
legs = fixes.generate_triplegs(stays).to_pandas()
stays = stays.to_pandas()
print(legs.columns.tolist(), stays.columns.tolist())

# Pick the busiest day (most triplegs) for the picture.
legs["day"] = pd.to_datetime(legs["started_at"], utc=True).dt.date
day = legs.day.value_counts().index[0]
d_legs = legs[legs.day == day]
t0 = pd.to_datetime(d_legs.started_at, utc=True).min() - pd.Timedelta("1h")
t1 = pd.to_datetime(d_legs.finished_at, utc=True).max() + pd.Timedelta("1h")
d_fix = tdf[(tdf.datetime >= t0) & (tdf.datetime <= t1)]
d_stay = stays[(pd.to_datetime(stays.started_at, utc=True) >= t0) & (pd.to_datetime(stays.started_at, utc=True) <= t1)]

# (file, figure size, legend font size, legend columns, legend height share): wide for desktop, tall for phones.
VARIANTS = [
    ("home-geolife-staypoints-triplegs.png", (6, 5), 11, 3, 0.07),
    ("home-geolife-staypoints-triplegs-mobile.png", (4, 4.8), 14, 1, 0.2),
]
for name, size, fs, ncol, legend_h in VARIANTS:
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.scatter(d_fix.lng, d_fix.lat, s=3, color="#6f6f6f", alpha=0.55, linewidths=0, label="position fixes")
    palette = plt.get_cmap("tab10")
    for i, (_, leg) in enumerate(d_legs.iterrows()):
        m = (d_fix.datetime >= pd.Timestamp(leg.started_at)) & (d_fix.datetime <= pd.Timestamp(leg.finished_at))
        seg = d_fix[m]
        ax.plot(seg.lng, seg.lat, lw=2, color=palette(i % 10), label="triplegs" if i == 0 else None)
    ax.scatter(
        d_stay.lng, d_stay.lat, s=90, facecolor="none", edgecolor=GOLD, linewidths=2, label="staypoints", zorder=5
    )
    ax.set_aspect("equal")
    # Square window centred on the stops; the rare long trip leaves the frame.
    cx, cy = d_stay.lng.median(), d_stay.lat.median()
    half = (
        max(
            d_stay.lng.quantile(0.9) - d_stay.lng.quantile(0.1),
            (d_stay.lat.quantile(0.9) - d_stay.lat.quantile(0.1)) / 0.77,
        )
        * 0.75
    )
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half * 0.77, cy + half * 0.77)
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        labelcolor=TEXT,
        fontsize=fs,
        loc="lower center",
        ncol=ncol,
        markerscale=1.4 if ncol == 1 else 1,
    )
    fig.tight_layout(pad=0.3, rect=(0, legend_h, 1, 1))
    fig.savefig(OUT / name, transparent=True)
    print("day", day, "triplegs", len(d_legs), "staypoints", len(d_stay), "fixes", len(d_fix))
