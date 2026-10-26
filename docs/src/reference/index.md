# API reference

Every public function and class in fastmob, grouped by module. Inputs can be any
pandas or Polars dataframe, and results come back in the same backend.

```python
import pandas as pd
import fastmob

df = pd.DataFrame({
    "uid": ["alice"] * 3,
    "datetime": pd.date_range("2020-01-01 08:00", periods=3, freq="h"),
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

| API | Description |
| --- | --- |
| [Data structures](data-structures.md) | Typed trajectory, hierarchy and flow wrappers. |
| [Preprocessing](preprocessing.md) | Filtering, segmentation, staypoints, H3 and OD preparation. |
| [Trajectory](trajectory.md) | Interpolation, smoothing, shape clustering and comparison. |
| [Network](network.md) | Road and rail graphs, routing, snapping and map matching. |
| [Integration](integration.md) | Joins with points of interest and events. |
| [Measures](measures/index.md) | Individual, collective, contact-network and evaluation metrics. |
| [Fitting](fitting.md) | Truncated power law, visitation law and daily-location fits. |
| [Models](models.md) | Mobility generation models. |
| [Input/output](io.md) | Readers and writers for common trajectory formats. |
| [Privacy](privacy.md) | Re-identification risk assessment for trajectories. |
| [Data](data.md) | Bundled and downloadable datasets. |
| [Visualization](visualization.md) | Interactive ECharts views (`fastmob[vis]`). |

!!! note "Column and backend conventions"

    Most APIs auto-detect `datetime`, `lat`, `lng`, and `uid` from common
    names. Pass explicit `*_col` arguments when your schema differs. See
    [Columns and backends](../learn/use-custom-columns-and-dataframe-backends.md)
    for the supported workflow.
