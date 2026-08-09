---
icon: lucide/circle-pause
---

# Staypoints

`Staypoints` represents intervals during which a person remained at a place.
It is normally created from `Positionfixes.generate_staypoints()`, and keeps the
metadata needed to resolve its user, coordinates, start time, and end time.

## Required data

Each row needs a start timestamp and, when validation is enabled, an end
timestamp no earlier than the start. Latitude and longitude are used by location
generation. `staypoint_id` is added during hierarchy generation and connects
staypoints to later trips.

```python
locations, assigned = staypoints.generate_user_locations(epsilon_km=0.1)
active = assigned.create_activity_flag(time_threshold_min=15)
```

Use user-scoped locations for personal recurring places, or
`generate_global_locations()` when multiple users need a shared catalogue.

## API

::: fastmob.core.staypoints_dataframe.Staypoints
    options:
      show_source: false
