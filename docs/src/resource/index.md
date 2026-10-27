# Resources

Datasets, packages, community and background reading for Fastkit-Mobility.

<div class="feature-grid">
<a class="feature-card" href="https://pypi.org/project/fastmob/">
<h3>PyPI →</h3>
<p><code>pip install fastmob</code></p>
</a>
<a class="feature-card" href="https://github.com/gefgu/fastmob">
<h3>Source code →</h3>
<p>Rust kernels, Python API and tests.</p>
</a>
<a class="feature-card" href="https://github.com/gefgu/fastmob/issues">
<h3>Get help →</h3>
<p>Report bugs and ask questions.</p>
</a>
</div>

## Datasets

- **[Brightkite](https://snap.stanford.edu/data/loc-brightkite.html)**: about 4M location check-ins from a social network. The workload for the [benchmarks](../features/benchmarks.md).
- **[GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)**: GPS trajectories of 182 users in Beijing. Read it with `load_geolife_trajectories`; used in the [stay-point notebook](../learn/notebooks/stay-point-detection-geolife.md).
- **Foursquare NYC and others**: datasets from scikit-mobility, available through `load_dataset`. See the [data reference](../reference/data.md).

Check each dataset's own page for its license and terms of use before redistributing it.

## Background reading

- [scikit-mobility](https://scikit-mobility.github.io/scikit-mobility/): the library whose measure API Fastkit-Mobility follows.
- [trackintel](https://trackintel.readthedocs.io/): the staypoint, tripleg, trip and tour hierarchy that inspired Fastkit-Mobility's.
- [Narwhals](https://narwhals-dev.github.io/narwhals/): the dataframe-agnostic layer that lets Fastkit-Mobility accept pandas and Polars.

### Papers

- Pappalardo, Simini, Barlacchi, Pellungrini. *scikit-mobility: A Python Library for the Analysis, Generation, and Risk Assessment of Mobility Data.* Journal of Statistical Software, 2022.
- Martin, Hong, Raubal and others. *Trackintel: An open-source Python library for human mobility analysis.* Computers, Environment and Urban Systems, 2023.
- Cho, Myers, Leskovec. *Friendship and mobility: user movement in location-based social networks.* KDD 2011 (the Brightkite data).
- Zheng, Zhang, Xie, Ma. *Mining interesting locations and travel sequences from GPS trajectories.* WWW 2009 (the GeoLife data).

## Terms used in these docs

Ordered along the hierarchy from raw fixes to tours.

<div class="figure">
<picture>
<source media="(max-width: 600px)" srcset="../assets/images/home/mobility-hierarchy-vertical.svg">
<img src="../assets/images/home/mobility-hierarchy.svg" alt="Positionfixes produce Staypoints and Triplegs, which combine into Trips and then Tours; Staypoints cluster into Locations, which tours return to.">
</picture>
</div>

| Class | Meaning |
| --- | --- |
| [`TrajDataFrame`](../reference/data-structures/traj-dataframe.md) | **Trajectory**: timestamped positions of one or many users |
| [`Positionfixes`](../reference/data-structures/positionfixes.md) | **Position fix**: one timestamped latitude/longitude observation |
| [`Staypoints`](../reference/data-structures/staypoints.md) | **Staypoint**: a period spent within a small spatial area |
| [`Locations`](../reference/data-structures/locations.md) | **Location**: a recurring place formed by clustering staypoints |
| [`Triplegs`](../reference/data-structures/triplegs.md) | **Tripleg**: movement between two stays |
| [`Trips`](../reference/data-structures/trips.md) | **Trip**: triplegs merged between two activity stays |
| [`Tours`](../reference/data-structures/tours.md) | **Tour**: consecutive trips that return to the same place |
| [`FlowDataFrame`](../reference/data-structures/flow-dataframe.md) | **OD flow**: a count or value from an origin to a destination |

!!! tip "Citing Fastkit-Mobility"

    See [Cite Fastkit-Mobility](../about/index.md#cite-fastkit-mobility) for a BibTeX entry.
