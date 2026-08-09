---
icon: lucide/rotate-ccw
---

# Tours

`Tours` represents a sequence of trips that starts and ends at the same
location. It is the highest aggregation level in the Trackintel-style hierarchy
and can contain multiple trips.

## Required data

Rows require `tour_id`, `started_at`, and `finished_at`. Generated tours also
store their anchor `location_id`, the list of trip IDs in `journey`, and a user
ID when applicable.

```python
tours = trips.generate_tours(staypoints_with_locations)
print(tours.df)
```

Tour generation requires location-assigned staypoints, typically produced by
`Staypoints.generate_user_locations()` or a validated global assignment.

## API

::: fastmob.core.tours_dataframe.Tours
    options:
      show_source: false
