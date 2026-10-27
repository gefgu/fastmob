---
icon: lucide/map-pin
---

# Positionfixes

`Positionfixes` is the raw-observation level of the Trackintel-style mobility
hierarchy. It is a semantic subclass of [`TrajDataFrame`](traj-dataframe.md):
one row represents one GPS fix with a timestamp, coordinates, and optionally a
user ID.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Positionfixes`](#fastmob.core.positionfixes_dataframe.Positionfixes) | Semantic alias for `TrajDataFrame` at the base of the Positionfixes -> Staypoints -> Triplegs -> Trips -> Tours hierarchy. |  |
| [`generate_staypoints`](#fastmob.core.positionfixes_dataframe.Positionfixes.generate_staypoints) | Detect stop locations, returning them as a typed `Staypoints` level. | [Build the visits](../../learn/motifs.md#build-the-visits) |
| [`generate_triplegs`](#fastmob.core.positionfixes_dataframe.Positionfixes.generate_triplegs) | Derive movement segments (triplegs) between staypoints. | [Movement between stops](../../learn/trajectory-hierarchy.md#movement-between-stops-triplegs-and-transport-mode) |
<!-- api-table:end -->

## Required data

Use the same trajectory columns as `TrajDataFrame`: datetime, latitude,
longitude, and optionally a user identifier. Calling `generate_staypoints()`
returns interval-based `Staypoints`; `generate_triplegs()` uses those detected
stops to derive movement summaries.

```python
from fastmob import Positionfixes

fixes = Positionfixes(traj, sort=True)
staypoints = fixes.generate_staypoints(minutes_for_a_stop=20)
triplegs = fixes.generate_triplegs(staypoints)
```

Continue with [`Staypoints`](staypoints.md) to assign recurring locations or
with [`Triplegs`](triplegs.md) to classify movement.

## API

::: fastmob.core.positionfixes_dataframe.Positionfixes
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Build the visits](../../learn/motifs.md#build-the-visits) · [Movement between stops](../../learn/trajectory-hierarchy.md#movement-between-stops-triplegs-and-transport-mode)

