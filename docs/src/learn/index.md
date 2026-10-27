---
hide:
  - toc
---

# Learn

Short, runnable guides from a first trajectory to fitted mobility laws. New to
Fastkit-Mobility? Start with the first guide, then follow the order below. Need every
parameter? Each guide links to the [API reference](../reference/index.md).

## Guides

<div class="how-to-grid">
<a class="how-to-card how-to-card--start" href="analyze-a-simple-trajectory.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-trajectory.png" alt="One GeoLife outing as a path coloured by time, with a dashed circle showing its radius of gyration">
<div>
<span class="how-to-card__eyebrow">Start here</span>
<h2>Start with a trajectory</h2>
<p>Build a small trajectory and compute your first mobility measures.</p>
<span class="how-to-card__meta">Data: synthetic (figure: GeoLife)</span>
</div>
</a>
<a class="how-to-card" href="use-custom-columns-and-dataframe-backends.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-backends.svg" alt="A pandas or Polars dataframe goes through fastmob and comes back as the same type">
<h2>Columns and backends</h2>
<p>Use your own column names, or move the same workflow between pandas and Polars.</p>
<span class="how-to-card__meta">Data: synthetic</span>
</a>
<a class="how-to-card" href="clean-and-segment.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-clean-segment.png" alt="The longest movement segments of one GeoLife user, each in its own colour">
<h2>Clean and segment trajectories</h2>
<p>Remove implausible jumps and split traces into movement episodes.</p>
<span class="how-to-card__meta">Data: GeoLife</span>
</a>
<a class="how-to-card" href="staypoints-and-locations.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-staypoints.png" alt="GeoLife GPS fixes in grey with detected staypoints as gold circles">
<h2>Create staypoints and locations</h2>
<p>Detect where people stop and cluster repeated visits into locations.</p>
<span class="how-to-card__meta">Data: GeoLife</span>
</a>
<a class="how-to-card" href="individual-and-collective-measures.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-measures.png" alt="Histogram of radius of gyration across Brightkite users">
<h2>Measure people and populations</h2>
<p>Quantify mobility per person and across a population.</p>
<span class="how-to-card__meta">Data: Brightkite</span>
</a>
<a class="how-to-card" href="network-analysis.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-route.svg" alt="Noisy GPS points snapped onto a road line">
<h2>Route and map-match trajectories</h2>
<p>Build network-aware workflows for routes and map-matched traces.</p>
<span class="how-to-card__meta">Data: road and rail networks</span>
</a>
<a class="how-to-card" href="fit-and-evaluate.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-fit.png" alt="Truncated power law fitted to Brightkite jump-length distribution on log-log axes">
<h2>Fit and evaluate mobility results</h2>
<p>Fit mobility laws and evaluate them against your data.</p>
<span class="how-to-card__meta">Data: Brightkite</span>
</a>
</div>

Also see [daily mobility motifs](motifs.md), a worked example of matching daily movement patterns.

## Notebook examples

<div class="how-to-grid how-to-grid--two">
<a class="how-to-card" href="notebooks/stay-point-detection-geolife.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-notebook-staypoints.png" alt="GeoLife GPS traces in grey with staypoints as gold circles sized by how long each stay lasted">
<h2>Stay-point detection notebook</h2>
<p>An end-to-end run on GeoLife GPS traces, compared with trackintel.</p>
<span class="how-to-card__meta">Data: GeoLife · Binder notebook</span>
</a>
<a class="how-to-card" href="compare-to-mobility-laws.md">
<img class="how-to-card__thumb" src="../assets/images/learn/learn-thumb-notebook-laws.png" alt="Brightkite jump-length histogram in grey with the fitted truncated power law in gold">
<h2>Compare mobility laws notebook</h2>
<p>Fit a power law to the distribution of Brightkite jump lengths.</p>
<span class="how-to-card__meta">Data: Brightkite · Binder notebook</span>
</a>
</div>
