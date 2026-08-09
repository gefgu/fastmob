---
icon: lucide/route
---

# Triplegs

`Triplegs` summarizes one movement segment between consecutive staypoints. Each
row describes a door-to-door segment rather than its raw point geometry, making
it suitable for mode classification and trip aggregation.

## Required data

Validated triplegs contain `tripleg_id`, `started_at`, `finished_at`,
`length_km`, and `duration_s`, plus a user ID when the data represents multiple
people. Generated triplegs also include mean speed information.

```python
classified = triplegs.predict_transport_mode()
trips = classified.generate_trips(active_staypoints)
```

`generate_trips()` requires staypoints with an `activity` flag. Create that flag
first with `Staypoints.create_activity_flag()`.

## API

::: fastmob.core.triplegs_dataframe.Triplegs
    options:
      show_source: false
