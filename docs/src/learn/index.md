---
hide:
  - toc
---

# Learn

Guides for mobility researchers, grouped by what you want to do. Start at chapter 1 if this is your first time. Each chapter mixes **notebooks** (runnable end to end in Colab or Binder, with real public data where possible) and **concept pages** (the ideas and choices, no code to run). Looking for a specific function? Use the [API reference](../reference/index.md).

## 1. Getting started

*What does mobility data look like, and how do I use it?*

- [**Start with a trajectory**](analyze-a-simple-trajectory.md): notebook, synthetic data. Build a trajectory table and compute your first four measures.
- [**Columns and backends**](use-custom-columns-and-dataframe-backends.md): use your own column names, or pandas and Polars interchangeably.

## 2. Preparing trajectories

*How do I turn noisy GPS fixes into trips, stops and places?*

- [**Cleaning and segmentation**](clean-and-segment.md): concept page. What to remove or split, and which choices change your results.
- [**Clean noisy GPS data and find stops**](clean-and-detect-stops.md): notebook, GeoLife. Filter glitches, split at gaps, detect stops, group them into places.
- [**Stops and locations**](staypoints-and-locations.md): concept page. Staypoints, locations and the positions-to-tours hierarchy.

## 3. Measuring mobility

*How far, how often and how predictably do people move?*

- [**Measuring people and populations**](individual-and-collective-measures.md): concept page. Which measure answers which question, and its units.
- [**Measure individual mobility**](measure-individual-mobility.md): notebook, Brightkite. Range, places, entropy, home and work for hundreds of users.

## 4. Spatial and network analysis

*What happens on the road network, and what do daily routines look like?*

- [**Route and map-match trajectories**](network-analysis.md): build a road graph, snap traces and query routes.
- [**Find daily mobility motifs**](motifs.md): match each day's movement to a library of daily patterns.

## 5. Modelling and evaluating mobility

*Which law or model describes my data, and how well?*

- [**Fit and evaluate mobility results**](fit-and-evaluate.md): choose the comparison that fits your question.
- [**Compare to mobility laws**](compare-to-mobility-laws.md): notebook, Brightkite. Fit a jump-length law and compare it with reference models.

## Advanced notebook

- [**Stay-point detection benchmark**](notebooks/stay-point-detection-geolife.md): full 25-million-fix GeoLife run, compared with trackintel. Downloads the whole 313 MB archive and takes a long time; start with the chapter 2 notebook instead.
