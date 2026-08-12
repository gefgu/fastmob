# API reference

The reference is the authoritative contract for Fastmob's public APIs. It
documents accepted dataframe backends, detected columns, return schemas,
optional dependencies, and behavior that matters at production scale.

## Start from your input

| You have | Start here |
| --- | --- |
| Raw GPS or check-in points | [Data structures](data-structures.md), then [preprocessing](preprocessing.md) |
| A cleaned trajectory | [Trajectory](trajectory.md) and [individual measures](measures/individual.md) |
| Visits, stays, or trips | [Collective measures](measures/collective.md) |
| A road/rail graph or GPS trace | [Network](network.md) |
| A model output to compare | [Evaluation](measures/evaluation.md) |

## Modules

- [Data structures](data-structures.md) — typed trajectory, hierarchy, and flow wrappers.
- [Preprocessing](preprocessing.md) — filtering, segmentation, stays, H3, and OD preparation.
- [Trajectory](trajectory.md) — interpolation, smoothing, shape clustering, and comparison.
- [Network](network.md) — Overture graphs, Arrow routing, snapping, and map matching.
- [Integration](integration.md) — joins with points of interest and events.
- [Measures](measures/index.md) — individual, collective, and evaluation metrics.
- [Models](models.md) — mobility generation models.
- [Input/output](io.md), [privacy](privacy.md), [data](data.md), and [visualization](visualization.md).

!!! note "Column and backend conventions"

    Most APIs auto-detect `datetime`, `lat`, `lng`, and `uid` from common
    names. Pass explicit `*_col` arguments when your schema differs. See
    [Columns and backends](../learn/use-custom-columns-and-dataframe-backends.md)
    for the supported workflow.
