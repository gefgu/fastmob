# API reference

Every public function and class in fastmob, grouped by module. Inputs can be any
pandas or Polars dataframe, and results come back in the same backend.

```python
import pandas as pd
import fastmob

df = pd.DataFrame({
    "uid": ["alice"] * 3,
    "datetime": pd.date_range(
        "2020-01-01 08:00", periods=3, freq="h"
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

| API | Description | Key functions |
| --- | --- | --- |
| [Data structures](data-structures.md) | Typed trajectory, hierarchy and flow wrappers. | `TrajDataFrame`, `Positionfixes` |
| [Preprocessing](preprocessing.md) | Filtering, segmentation, staypoints, H3 and OD preparation. | `filter`, `segment`, `stay_locations` |
| [Trajectory](trajectory.md) | Interpolation, smoothing, shape clustering and comparison. | `interpolate`, `smooth` |
| [Network](network.md) | Road and rail graphs, routing, snapping and map matching. | `match_trajectory`, `build_road_graph` |
| [Integration](integration.md) | Joins with points of interest and events. | `join_with_pois`, `join_with_events` |
| [Measures](measures/index.md) | Individual, collective, contact-network and evaluation metrics. | `radius_of_gyration`, `jump_lengths` |
| [Fitting](fitting.md) | Truncated power law, visitation law and daily-location fits. | `fit_values_to_truncated_powerlaw` |
| [Models](models.md) | Mobility generation models. | `Gravity`, `SpatialEPR` |
| [Input/output](io.md) | Readers and writers for common trajectory formats. | `read`, `load_geolife_trajectories` |
| [Privacy](privacy.md) | Re-identification risk assessment for trajectories. | `location_risk` |
| [Data](data.md) | Bundled and downloadable datasets. | `load_dataset` |
| [Visualization](visualization.md) | Interactive ECharts views (`fastmob[vis]`). | `fastmob.vis` |

!!! note "Column and backend conventions"

    Most APIs auto-detect `datetime`, `lat`, `lng`, and `uid` from common
    names. Pass explicit `*_col` arguments when your schema differs. See
    [Columns and backends](../learn/use-custom-columns-and-dataframe-backends.md)
    for the supported workflow.
