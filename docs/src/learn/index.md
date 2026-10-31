---
hide:
  - toc
---

# Learn

Guides for mobility researchers, grouped by what you want to do. Start at chapter 1 if this is your first time. Each chapter mixes **notebooks** (run end to end in Colab or Binder from a clean environment, with real public data wherever possible, each stating its data source and download size) and **concept pages** (the ideas and the choices that change your results, with no code to run). Looking for a specific function? Use the [API reference](../reference/index.md); each of its entries links to the notebook that runs it.

## 1. Getting started

*What does mobility data look like, and how do I use it?*

- [**Start with a trajectory**](analyze-a-simple-trajectory.md): notebook, synthetic data. Build a trajectory table and compute your first four measures.
- [**Use your own columns and dataframe library**](use-custom-columns-and-dataframe-backends.md): notebook, synthetic data. Non-standard column names, and pandas, Polars or PyArrow in and out.

## 2. Preparing trajectories

*How do I turn noisy GPS fixes into trips, stops and places?*

- [**Cleaning and segmentation**](clean-and-segment.md): concept page. What to remove or split, and which choices change your results.
- [**Clean noisy GPS data and find stops**](clean-and-detect-stops.md): notebook, GeoLife. Filter glitches, split at gaps, thin the data, detect stops, group them into places.
- [**Fill gaps, smooth, compare and convert trajectories**](process-trajectories.md): notebook, GeoLife. Local time, filling recording gaps, smoothing, trajectory distance, grids and tilers, shape clustering, GeoPandas, file round trip.
- [**Stops and locations**](staypoints-and-locations.md): concept page. Staypoints, locations and the positions-to-tours hierarchy.
- [**Build the trajectory hierarchy: stops, trips and tours**](trajectory-hierarchy.md): notebook, GeoLife. Staypoints, places, home and work, triplegs, transport mode, trips and tours.

## 3. Measuring mobility

*How far, how often and how predictably do people move, and where does the population go?*

- [**Measuring people and populations**](individual-and-collective-measures.md): concept page. Which measure answers which question, and its units.
- [**Measure individual mobility**](measure-individual-mobility.md): notebook, Brightkite. Range, places, entropy, home and work for hundreds of users.
- [**Routine, predictability and mobility profiles**](visit-based-measures.md): notebook, GeoLife. Regularity, entropy, predictability, return and exploration, activity profiles.
- [**Find daily mobility motifs**](motifs.md): notebook, GeoLife. The shape of each person's day.
- [**Population measures**](population-measures.md): notebook, Brightkite. Homes and visits per place, spread over time, diversity of visitors.
- [**Flows between places**](flows-between-places.md): notebook, Brightkite. Origin-destination flows, zones, and comparing two flow tables.
- [**Occupancy and phone records**](occupancy-and-phone-records.md): notebook, Brightkite and synthetic call records. When places are busy, shared visitors, stays and trips from phone records.

## 4. Spatial and network analysis

*What happens on the road network, and who meets whom?*

- [**Roads, rails and map matching**](network-analysis.md): concept page. Graphs, snapping, map matching and route flows.
- [**Road networks: matching, routing and route flows**](road-network-analysis.md): notebook, GeoLife and Overture Maps. Match a trace, route, compare road and straight-line distance, join points of interest and events.
- [**Contact networks from stays**](contact-networks.md): notebook, Brightkite. Co-presence contacts, structure against a random baseline, Friends, Bridges, Acquaintances.

## 5. Modelling, evaluating and protecting

*Which law or model describes my data, how well, and how identifiable are the people in it?*

- [**Compare your data to mobility laws**](compare-to-mobility-laws.md): notebook, Brightkite. Jump lengths, the visitation law and daily places.
- [**How the fits work**](fitting-methods.md): concept page. The estimators behind the distance and gravity fits, their uncertainty and what to report.
- [**Validate against census flows**](validate-against-census-flows.md): notebook, synthetic reference table and Brightkite. Home-to-work flows from trajectories, expansion weights, gravity distance decay with a confidence interval, gravity against radiation.
- [**Validate mobility with GeoLife**](validate-mobility-geolife.md): notebook, GeoLife GPS tracks. The full cycle from raw tracks to a distance law with confidence intervals, a gravity distance-decay parameter, and a held-out gravity against radiation comparison.
- [**Generate mobility with models**](generate-mobility.md): notebook, synthetic data. Exploration and return models, diaries, social ties, gravity and radiation flows.
- [**Fit and compare distributions**](fit-and-evaluate.md): notebook, Brightkite. Error, information, distribution and common-part metrics.
- [**Privacy risk of trajectories**](privacy-risk.md): notebook, Brightkite. Re-identification risk under eight attacker models.

## 6. Charting results

*How do I draw what I measured?*

- [**Chart your results**](visualize-results.md): notebook, Brightkite and GeoLife. Distributions, bars, boxes, scatters and heatmaps from any result, and the charts for activities, mobility profiles, motifs and fitted laws.
