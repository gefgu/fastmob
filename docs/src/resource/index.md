# Resources

Datasets, packages, community and background reading for fastmob.

<div class="feature-grid">
<a class="feature-card" href="https://pypi.org/project/fastmob/">
<h3>PyPI →</h3>
<p>Install with <code>pip install fastmob</code>; prebuilt wheels for Linux, macOS and Windows.</p>
</a>
<a class="feature-card" href="https://github.com/gefgu/fastmob">
<h3>Source code →</h3>
<p>Rust kernels, Python API, tests and benchmark scripts.</p>
</a>
<a class="feature-card" href="https://github.com/gefgu/fastmob/issues">
<h3>Get help →</h3>
<p>Report bugs and ask questions on the GitHub issue tracker.</p>
</a>
</div>

## Datasets

- **[Brightkite](https://snap.stanford.edu/data/loc-brightkite.html)**: about 4M location check-ins from a social network. The workload for the [benchmarks](../features/benchmarks.md).
- **[GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)**: GPS trajectories of 182 users in Beijing. Read it with `load_geolife_trajectories`; used in the [stay-point notebook](../learn/notebooks/stay-point-detection-geolife.md).
- **Foursquare NYC and others**: datasets from scikit-mobility, available through `load_dataset`. See the [data reference](../reference/data.md).

## Background reading

- [scikit-mobility](https://scikit-mobility.github.io/scikit-mobility/): the library whose measure API fastmob follows.
- [trackintel](https://trackintel.readthedocs.io/): the staypoint, tripleg, trip and tour hierarchy that inspired fastmob's.
- [Narwhals](https://narwhals-dev.github.io/narwhals/): the dataframe-agnostic layer that lets fastmob accept pandas and Polars.

## Terms used in these docs

Ordered along the hierarchy from raw fixes to tours.

<div class="figure-scroll">
<img src="../assets/images/home/mobility-hierarchy.svg" alt="Positionfixes produce Staypoints and Triplegs, which combine into Trips and then Tours; Staypoints cluster into Locations, which tours return to.">
</div>

| Term | Meaning |
| --- | --- |
| Trajectory: [`TrajDataFrame`](../reference/data-structures/traj-dataframe.md) | Timestamped positions of one or many users |
| Position fix: [`Positionfixes`](../reference/data-structures/positionfixes.md) | One timestamped latitude/longitude observation |
| Staypoint: [`Staypoints`](../reference/data-structures/staypoints.md) | A period spent within a small spatial area |
| Location: [`Locations`](../reference/data-structures/locations.md) | A recurring place formed by clustering staypoints |
| Tripleg: [`Triplegs`](../reference/data-structures/triplegs.md) | Movement between two stays |
| Trip: [`Trips`](../reference/data-structures/trips.md) | Triplegs merged between two activity stays |
| Tour: [`Tours`](../reference/data-structures/tours.md) | Consecutive trips that return to the same place |
| OD flow: [`FlowDataFrame`](../reference/data-structures/flow-dataframe.md) | A count or value from an origin to a destination |

!!! tip "Citing fastmob"

    See [Cite fastmob](../about/index.md#cite-fastmob) for a BibTeX entry.
