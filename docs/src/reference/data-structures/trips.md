---
icon: lucide/car
---

# Trips

`Trips` aggregates consecutive triplegs between activity staypoints. It captures
the full journey between meaningful activities, including the contributing
tripleg IDs and endpoint staypoint IDs.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Trips`](#fastmob.core.trips_dataframe.Trips) | One row per trip. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`Trips.common_part_of_commuters`](#fastmob.core.trips_dataframe.Trips.common_part_of_commuters) | Compare trips using globally comparable endpoint location IDs. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`Trips.common_part_of_commuters_distance`](#fastmob.core.trips_dataframe.Trips.common_part_of_commuters_distance) | Return the distance-weighted CPC score against another trip set. | [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) |
| [`Trips.common_part_of_links`](#fastmob.core.trips_dataframe.Trips.common_part_of_links) | Return the common part of links score against another trip set. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`Trips.from_triplegs`](#fastmob.core.trips_dataframe.Trips.from_triplegs) | Derive trips from triplegs + activity-flagged staypoints. | [The same pipeline, step by step](../../learn/trajectory-hierarchy.md#the-same-pipeline-step-by-step) |
| [`Trips.generate_tours`](#fastmob.core.trips_dataframe.Trips.generate_tours) | Group consecutive trips into tours (round trips back to the same location). | [Trips and tours](../../learn/trajectory-hierarchy.md#trips-and-tours) |
| [`Trips.to_flow_dataframe`](#fastmob.core.trips_dataframe.Trips.to_flow_dataframe) | Materialize this trip set as aggregated sparse OD flows. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
<!-- api-table:end -->

## Required data

Rows require `trip_id`, `started_at`, and `finished_at`. Generated trips include
`origin_staypoint_id`, `destination_staypoint_id`, and `tripleg_ids`. When their
staypoints have global location IDs, trips also carry endpoint location IDs for
OD comparison and flow aggregation.

```python
flows = trips.to_flow_dataframe()
score = trips.common_part_of_commuters(other_trips)
tours = trips.generate_tours(staypoints_with_locations)
```

Use [`FlowDataFrame`](flow-dataframe.md) for a materialized sparse OD table or
[`Tours`](tours.md) to group journeys that return to their starting place.

## API

::: fastmob.core.trips_dataframe.Trips
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) · [Common part of commuters and links](../../learn/fit-and-evaluate.md#common-part-of-commuters-and-links) · [The same pipeline, step by step](../../learn/trajectory-hierarchy.md#the-same-pipeline-step-by-step) · [Trips and tours](../../learn/trajectory-hierarchy.md#trips-and-tours)

