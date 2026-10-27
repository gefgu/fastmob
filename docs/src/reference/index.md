# API reference

Every public function and class in Fastkit-Mobility, grouped by module. Inputs can be any
pandas or Polars dataframe, and results come back in the same backend.

```python
import pandas as pd
import fastmob

df = pd.DataFrame({
    "uid": ["alice"] * 3,
    "datetime": pd.date_range(
        "2020-01-01", periods=3
    ),
    "lat": [41.9, 42.0, 42.1],
    "lng": [12.5, 12.5, 12.6],
})
fastmob.radius_of_gyration(df)
```

```text
     uid  radius_of_gyration
0  alice            9.878827
```

The result is a pandas DataFrame because `df` is pandas; pass Polars and you get Polars.

## Start from your input

| You have | Start here |
| --- | --- |
| Raw GPS or check-in points | [Data structures](data-structures.md), then [preprocessing](preprocessing.md) |
| A cleaned trajectory | [Trajectory](trajectory.md) and [individual measures](measures/individual.md) |
| Visits, stays, or trips | [Collective measures](measures/collective.md) |
| A road/rail graph or GPS trace | [Network](network.md) |
| A model output to compare | [Evaluation](measures/evaluation.md) |

## Modules

| Module |
| --- |
| [Data structures](data-structures.md)<br><span class="muted">Typed trajectory, hierarchy and flow wrappers.</span><br>`TrajDataFrame`, `Positionfixes` |
| [Preprocessing](preprocessing.md)<br><span class="muted">Filtering, segmentation, staypoints, H3 and OD preparation.</span><br>`filter`, `segment`, `stay_locations` |
| [Trajectory](trajectory.md)<br><span class="muted">Interpolation, smoothing, shape clustering and comparison.</span><br>`interpolate`, `smooth` |
| [Network](network.md)<br><span class="muted">Road and rail graphs, routing, snapping and map matching.</span><br>`match_trajectory`, `build_road_graph` |
| [Integration](integration.md)<br><span class="muted">Joins with points of interest and events.</span><br>`join_with_pois`, `join_with_events` |
| [Measures](measures/index.md)<br><span class="muted">Individual, collective, contact-network and evaluation metrics.</span><br>`radius_of_gyration`, `jump_lengths` |
| [Fitting](fitting.md)<br><span class="muted">Truncated power law, visitation law and daily-location fits.</span><br>`fit_values_to_truncated_powerlaw` |
| [Models](models.md)<br><span class="muted">Mobility generation models.</span><br>`Gravity`, `SpatialEPR` |
| [Input/output](io.md)<br><span class="muted">Readers and writers for common trajectory formats.</span><br>`read`, `load_geolife_trajectories` |
| [Privacy](privacy.md)<br><span class="muted">Re-identification risk assessment for trajectories.</span><br>`location_risk` |
| [Data](data.md)<br><span class="muted">Bundled and downloadable datasets.</span><br>`load_dataset` |
| [Visualization](visualization.md)<br><span class="muted">Interactive ECharts views (`fastmob[vis]`).</span><br>`fastmob.vis` |

!!! note "Column and backend conventions"

    Most APIs auto-detect `datetime`, `lat`, `lng`, and `uid` from common
    names. Pass explicit `*_col` arguments when your schema differs. See
    [Columns and backends](../learn/use-custom-columns-and-dataframe-backends.md)
    for the supported workflow.
