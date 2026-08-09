---
icon: lucide/map-pin
---

# Positionfixes

`Positionfixes` is the raw-observation level of the Trackintel-style mobility
hierarchy. It is a semantic subclass of [`TrajDataFrame`](traj-dataframe.md):
one row represents one GPS fix with a timestamp, coordinates, and optionally a
user ID.

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
