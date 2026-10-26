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

| Dataset | What it is | How to use it in fastmob |
| --- | --- | --- |
| [Brightkite](https://snap.stanford.edu/data/loc-brightkite.html) | About 4M location check-ins from a social network | Benchmark workload for the [benchmarks page](../features/benchmarks.md) |
| [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) | GPS trajectories of 182 users in Beijing | `load_geolife_trajectories`; used in the [stay-point notebook](../learn/notebooks/stay-point-detection-geolife.md) |
| Foursquare NYC and others | Datasets from scikit-mobility | `load_dataset`; see the [data reference](../reference/data.md) |

## Background reading

- [scikit-mobility](https://scikit-mobility.github.io/scikit-mobility/): the library whose measure API fastmob follows.
- [trackintel](https://trackintel.readthedocs.io/): the staypoint, tripleg, trip and tour hierarchy that inspired fastmob's.
- [Narwhals](https://narwhals-dev.github.io/narwhals/): the dataframe-agnostic layer that lets fastmob accept pandas and Polars.

## Terms used in these docs

Ordered along the hierarchy from raw fixes to tours. See [Data structures](../reference/data-structures.md) for the full picture.

| Term | Meaning | Class |
| --- | --- | --- |
| Trajectory | Timestamped positions of one or many users | [`TrajDataFrame`](../reference/data-structures/traj-dataframe.md) |
| Position fix | One timestamped latitude/longitude observation | [`Positionfixes`](../reference/data-structures/positionfixes.md) |
| Staypoint | A period spent within a small spatial area | [`Staypoints`](../reference/data-structures/staypoints.md) |
| Tripleg | Movement between two stays | [`Triplegs`](../reference/data-structures/triplegs.md) |
| Trip | Triplegs merged between two activity stays | [`Trips`](../reference/data-structures/trips.md) |
| Tour | Consecutive trips that return to the same place | [`Tours`](../reference/data-structures/tours.md) |
| Location | A recurring place formed by clustering staypoints | [`Locations`](../reference/data-structures/locations.md) |
| OD flow | A count or value from an origin to a destination | [`FlowDataFrame`](../reference/data-structures/flow-dataframe.md) |

To cite fastmob, see [About](../about/index.md#cite-fastmob).
