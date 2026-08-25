---
icon: lucide/waypoints
---

# Data Structures

fastmob uses small, typed wrappers around backend-native DataFrames. They retain
the underlying pandas, Polars, or other Narwhals-compatible frame while making
the columns and operations for a mobility-analysis level explicit.

## Choose a structure

| Structure | Use it for | Next step |
| --- | --- | --- |
| [`TrajDataFrame`](data-structures/traj-dataframe.md) | Generic timestamped trajectory points and the original fastmob trajectory API. | Measure, clean, map, or convert trajectories. |
| [`Positionfixes`](data-structures/positionfixes.md) | Raw GPS fixes in the Trackintel-style hierarchy. | Generate staypoints and triplegs. |
| [`Staypoints`](data-structures/staypoints.md) | Intervals where a person remained at a place. | Assign activity flags and locations. |
| [`Locations`](data-structures/locations.md) | Recurring user places or a shared global location catalogue. | Identify purposes or validate assignments. |
| [`Triplegs`](data-structures/triplegs.md) | Single movement segments between staypoints. | Predict mode or aggregate into trips. |
| [`Trips`](data-structures/trips.md) | Connected triplegs between activity staypoints. | Compare OD demand, create flows, or generate tours. |
| [`Tours`](data-structures/tours.md) | Round trips that return to their starting location. | Analyze journeys as a whole. |
| [`FlowDataFrame`](data-structures/flow-dataframe.md) | Sparse origin-destination flows. | Query, compare, or convert the OD matrix. |

## Mobility hierarchy

The hierarchy follows the Trackintel vocabulary. Start from raw
[`Positionfixes`](data-structures/positionfixes.md), detect
[`Staypoints`](data-structures/staypoints.md) and
[`Triplegs`](data-structures/triplegs.md), then aggregate them into
[`Locations`](data-structures/locations.md), [`Trips`](data-structures/trips.md),
and [`Tours`](data-structures/tours.md). A trip collection with global location
IDs can also become a [`FlowDataFrame`](data-structures/flow-dataframe.md).

`TrajDataFrame` remains the general-purpose trajectory wrapper. `Positionfixes`
is its semantic, hierarchy-aware subclass, so existing trajectory workflows can
adopt the hierarchy incrementally.

## Shared behavior

Every wrapper exposes `.df` for its original DataFrame and `.to_native()` to
return the native backend object. Use `.to_pandas()` or `.to_polars()` when a
specific native dataframe backend is required. Hierarchy wrappers also inherit
comparison and chart helpers from `BaseDataFrame`; the generated API on each
page documents the operations defined for that structure.
