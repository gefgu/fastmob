---
icon: lucide/map-pinned
---

# Locations

`Locations` is a catalogue of meaningful places. A catalogue is either
user-scoped, where a location ID is meaningful only for one user, or global,
where every user shares the same location IDs. This distinction makes OD and
collective analysis explicit.

## Required data

Rows require `location_id`, `center_lat`, and `center_lng`. User-scoped catalogues
also carry a user-ID column; global catalogues must not. The `scope` and `scheme`
metadata describe identity and whether IDs came from clustering, H3, or an
external source.

```python
locations, assigned = staypoints.generate_user_locations(epsilon_km=0.1)
labelled = locations.identify(assigned)
```

Global catalogues validate assignments used by models and collective measures;
user catalogues can be labeled as home, work, or other.

## API

::: fastmob.core.locations_dataframe.Locations
    options:
      show_source: false
