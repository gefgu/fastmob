"""Generate static figures for the docs home page.

Benchmark numbers are copied from the published 4M-row Brightkite chart
(docs/src/assets/benchmarks/fastmob_vs_skmob_individual_pandas_4M.png,
pandas backend, average seconds over 5 iterations).

Usage: uv run --with matplotlib python scripts/make_docs_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("docs/src/assets/images/home")
OUT.mkdir(parents=True, exist_ok=True)

# measure: (skmob seconds, fastmob seconds, published speedup) at 4M rows
BENCH = {
    "radius of gyration": (31.42, 0.100, 315.19),
    "jump lengths": (42.79, 0.131, 325.92),
    "location frequency": (115.29, 0.087, 1329.29),
    "home location": (76.97, 0.121, 637.66),
    "max distance from home": (182.82, 0.151, 1208.46),
}

GOLD, GREY, TEXT = "#a98600", "#8a8a8a", "#777777"

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
names = list(BENCH)[::-1]
for i, name in enumerate(names):
    sk, fm, speedup = BENCH[name]
    ax.barh(i + 0.18, sk, height=0.34, color=GREY)
    ax.barh(i - 0.18, fm, height=0.34, color=GOLD)
    ax.text(sk * 1.15, i + 0.18, f"{sk:.1f}s", va="center", fontsize=11, color=TEXT)
    ax.text(
        fm * 1.15, i - 0.18, f"{fm:.2f}s  ({speedup:.0f}x faster)",
        va="center", fontsize=11, color=GOLD, fontweight="bold",
    )
ax.set_yticks(range(len(names)), names, fontsize=12, color=TEXT)
ax.set_xlabel("Execution time on 4M Brightkite check-ins (seconds, log scale, lower is better)",
              fontsize=11, color=TEXT)
ax.tick_params(colors=TEXT, labelsize=10)
ax.set_xscale("log")
ax.set_xlim(0.05, 2000)
ax.set_ylim(-0.6, len(names) - 0.2)
ax.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(TEXT)
ax.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=GOLD)],
    labels=["scikit-mobility", "fastmob (pandas)"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
    frameon=False, labelcolor=TEXT, fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT / "home-benchmark-speedup.png", transparent=True)


# Portrait variant for phones: speedup factors only, large type.
fig, ax = plt.subplots(figsize=(4.0, 4.6), dpi=200)
items = sorted(BENCH.items(), key=lambda kv: kv[1][2])
for i, (name, (sk, fm, speedup)) in enumerate(items):
    ax.barh(i, speedup, height=0.55, color=GOLD)
    ax.text(speedup * 1.12, i, f"{speedup:.0f}x", va="center", fontsize=14,
            color=GOLD, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(100, 6000)
ax.set_yticks(range(len(items)), [n for n, _ in items], fontsize=12, color=TEXT)
ax.set_xticks([100, 1000], ["100x", "1000x"], fontsize=11, color=TEXT)
fig.suptitle("Speedup vs scikit-mobility", fontsize=12, color=TEXT)
ax.set_title("4M Brightkite check-ins", fontsize=10, color=TEXT, pad=6)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(TEXT)
ax.tick_params(colors=TEXT)
fig.tight_layout()
fig.savefig(OUT / "home-benchmark-speedup-mobile.png", transparent=True, bbox_inches="tight", pad_inches=0.15)
