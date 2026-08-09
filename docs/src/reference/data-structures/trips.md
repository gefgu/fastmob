---
icon: lucide/car
---

# Trips

`Trips` aggregates consecutive triplegs between activity staypoints. It captures
the full journey between meaningful activities, including the contributing
tripleg IDs and endpoint staypoint IDs.

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
