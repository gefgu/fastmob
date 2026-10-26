"""Generate the Learn card thumbnails from real fastmob output.

Needs the cached datasets in tests/shared/data (GeoLife, Brightkite).
Usage: uv run --with matplotlib python scripts/make_learn_thumbnails.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fastmob
from fastmob import Positionfixes, jump_lengths, radius_of_gyration

OUT = Path("docs/src/assets/images/learn")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("tests/shared/data")
GOLD, GREY, INK = "#c9a400", "#8a8a8a", "#8a8a8a"


def new_fig():
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    return fig, ax


def finish(fig, ax, name, axes=False):
    if not axes:
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)


def geolife_user(user="000"):
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "Geolife Trajectories 1.3" / "Data"
    tmp.mkdir(parents=True)
    shutil.copytree(DATA / "Geolife Trajectories 1.3" / "Data" / user, tmp / user)
    return fastmob.io.load_geolife_trajectories(str(tmp.parent)).to_pandas()


def staypoints_and_segments():
    tdf = geolife_user()
    tdf["datetime"] = pd.to_datetime(tdf["datetime"], utc=True)
    tdf = tdf.sort_values("datetime")
    # Keep the dense Beijing core so the picture is readable.
    tdf = tdf[(tdf.lat.between(39.9, 40.1)) & (tdf.lng.between(116.2, 116.5))]
    stays = Positionfixes(tdf).generate_staypoints(
        minutes_for_a_stop=20, spatial_radius_km=0.2
    ).to_pandas()

    fig, ax = new_fig()
    ax.scatter(tdf.lng, tdf.lat, s=1.5, color=GREY, alpha=0.5, linewidths=0)
    ax.scatter(stays.lng, stays.lat, s=40, facecolor="none", edgecolor=GOLD, linewidths=1.6)
    finish(fig, ax, "learn-thumb-staypoints.png")
    print("staypoints:", len(stays), "of", len(tdf), "fixes")

    from fastmob.preprocessing import filter as fm_filter, segment

    seg = segment(tdf, method="observation_gap", gap_s=30 * 60).to_pandas() \
        if hasattr(segment(tdf.head(10), method="observation_gap", gap_s=1800), "to_pandas") \
        else segment(tdf, method="observation_gap", gap_s=30 * 60)
    fig, ax = new_fig()
    ids = seg["segment_id"].to_numpy()
    uniq = pd.unique(ids)
    sizes = pd.Series(ids).value_counts()
    keep = list(sizes.index[:7])  # the longest movement episodes
    palette = [GOLD, "#4c78a8", "#e45756", "#54a24b", "#b279a2", "#f58518", "#72b7b2"]
    for i, sid in enumerate(keep):
        s = seg[ids == sid]
        ax.plot(s.lng, s.lat, lw=2, color=palette[i])
    finish(fig, ax, "learn-thumb-clean-segment.png")
    print("segments:", len(uniq))


def measures():
    df = pd.read_csv(
        DATA / "loc-brightkite_totalCheckins.txt.gz", sep="\t", header=None,
        names=["uid", "datetime", "lat", "lng", "loc"], nrows=600_000,
    ).dropna(subset=["lat", "lng"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    rg = radius_of_gyration(df)["radius_of_gyration"].to_numpy()
    rg = rg[rg > 0]
    fig, ax = new_fig()
    ax.hist(np.log10(rg), bins=40, color=GOLD)
    ax.set_xlim(-2.2, 3.8)
    ax.set_xticks([-2, -1, 0, 1, 2, 3], ["0.01", "0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("radius of gyration (km)", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    finish(fig, ax, "learn-thumb-measures.png", axes=True)
    print("users:", len(rg))


def fit():
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw
    from fastmob.measures.fitting.truncated_powerlaw import log_truncated_powerlaw

    df = pd.read_csv(
        DATA / "loc-brightkite_totalCheckins.txt.gz", sep="\t", header=None,
        names=["uid", "datetime", "lat", "lng", "loc"], nrows=600_000,
    ).dropna(subset=["lat", "lng"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    jl = jump_lengths(df)["jump_lengths"]
    values = np.concatenate([np.asarray(x) for x in jl])
    values = values[values > 0]
    (c, r0, beta, kappa), x, y = fit_values_to_truncated_powerlaw(values)
    fig, ax = new_fig()
    ax.loglog(x, y, "o", ms=4, color=GREY, label="Brightkite jump lengths")
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ax.loglog(xs, np.exp(log_truncated_powerlaw(xs, c, r0, beta, kappa)), color=GOLD, lw=2.5,
              label=f"truncated power law (beta={beta:.2f})")
    ax.set_xlabel("jump length (km)", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
    finish(fig, ax, "learn-thumb-fit.png", axes=True)
    print("fit beta", beta, "n", len(values))


def trajectory():
    """One GeoLife outing: path coloured by time, with its radius of gyration."""
    tdf = geolife_user()
    tdf["datetime"] = pd.to_datetime(tdf["datetime"], utc=True)
    tdf = tdf.sort_values("datetime")
    tdf = tdf[(tdf.lat.between(39.9, 40.1)) & (tdf.lng.between(116.2, 116.5))]
    day = tdf[tdf.datetime.dt.date == tdf.datetime.dt.date.iloc[len(tdf) // 2]].iloc[::8]
    rg = radius_of_gyration(day)["radius_of_gyration"].iloc[0]
    fig, ax = new_fig()
    ax.scatter(day.lng, day.lat, c=np.arange(len(day)), cmap="YlOrBr", s=14, zorder=3)
    ax.plot(day.lng, day.lat, color=GREY, lw=1, zorder=2)
    cx, cy = day.lng.mean(), day.lat.mean()
    ax.add_patch(plt.Circle((cx, cy), rg / 111.19, fill=False, ls="--", color=GOLD, lw=2))
    ax.set_aspect("equal")
    ax.text(0.02, 0.97, f"radius of gyration: {rg:.1f} km", transform=ax.transAxes,
            color=INK, fontsize=14, va="top", fontweight="bold")
    finish(fig, ax, "learn-thumb-trajectory.png")
    print("trajectory rg", rg, "points", len(day))


if __name__ == "__main__":
    fit()
    trajectory()
    staypoints_and_segments()
    measures()
