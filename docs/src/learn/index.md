---
hide:
  - toc
---

# Learn

Short, runnable guides from a first trajectory to fitted mobility laws. New to
fastmob? Start with the first guide, then follow the order below. Need every
parameter? Each guide links to the [API reference](../reference/index.md).

## Guides

<div class="how-to-grid">
<a class="how-to-card how-to-card--start" href="analyze-a-simple-trajectory.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-trajectory.png" alt="Two short trajectories with jump lengths and a radius-of-gyration circle">
<span class="how-to-card__eyebrow">Start here</span>
<h2>Start with a trajectory</h2>
<p>Build a small trajectory and compute your first mobility measures.</p>
<span class="how-to-card__meta">Synthetic data</span>
</a>
<a class="how-to-card" href="use-custom-columns-and-dataframe-backends.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-backends.svg" alt="A pandas or Polars dataframe goes through fastmob and comes back as the same type">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Columns and backends</h2>
<p>Use your own column names, or move the same workflow between pandas and Polars.</p>
<span class="how-to-card__meta">Synthetic data</span>
</a>
<a class="how-to-card" href="clean-and-segment.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-clean-segment.png" alt="The longest movement segments of one GeoLife user, each in its own colour">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Clean and segment trajectories</h2>
<p>Remove implausible jumps and split traces into movement episodes.</p>
<span class="how-to-card__meta">Synthetic data</span>
</a>
<a class="how-to-card" href="staypoints-and-locations.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-staypoints.png" alt="GeoLife GPS fixes in grey with detected staypoints as gold circles">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Create staypoints and locations</h2>
<p>Detect where people stop and cluster repeated visits into locations.</p>
<span class="how-to-card__meta">GeoLife</span>
</a>
<a class="how-to-card" href="individual-and-collective-measures.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-measures.png" alt="Histogram of radius of gyration across Brightkite users">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Measure people and populations</h2>
<p>Quantify mobility per person and across a population.</p>
<span class="how-to-card__meta">Brightkite</span>
</a>
<a class="how-to-card" href="network-analysis.md">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Route and map-match trajectories</h2>
<p>Build network-aware workflows for routes and map-matched traces.</p>
</a>
<a class="how-to-card" href="fit-and-evaluate.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-fit.png" alt="Truncated power law fitted to Brightkite jump-length distribution on log-log axes">
<span class="how-to-card__eyebrow">Guide</span>
<h2>Fit and evaluate mobility results</h2>
<p>Fit mobility laws and evaluate them against your data.</p>
<span class="how-to-card__meta">Brightkite</span>
</a>
</div>

## Notebook examples

<div class="how-to-grid">
<a class="how-to-card" href="notebooks/stay-point-detection-geolife.md">
<span class="how-to-card__eyebrow">Notebook</span>
<h2>Stay-point detection notebook</h2>
<p>An end-to-end run on GeoLife GPS traces, compared with trackintel.</p>
<span class="how-to-card__meta">Binder · GeoLife</span>
</a>
<a class="how-to-card" href="comparision-to-mobility-laws.md">
<span class="how-to-card__eyebrow">Notebook</span>
<h2>Compare mobility laws notebook</h2>
<p>Fit a power law to the distribution of Brightkite jump lengths.</p>
<span class="how-to-card__meta">Binder</span>
</a>
</div>
