# Integration

PyMove-style spatial and spatiotemporal joins: augment a trajectory
dataframe with columns describing its nearest point(s) of interest or
nearest event, rather than aggregating the trajectory into a per-user
summary like the measures under `fastmob.measures`.

| API | Description |
| --- | --- |
| [`join_with_pois`](#fastmob.integration.join_with_pois) | Join each trajectory point with its single nearest point of interest. |
| [`join_with_pois_by_category`](#fastmob.integration.join_with_pois_by_category) | Join each trajectory point with its nearest POI in each category. |
| [`join_with_events`](#fastmob.integration.join_with_events) | Join each trajectory point with the nearest event within a time window. |

Requires the `ai` extra (`pip install fastmob[ai]`, needs `scikit-learn`)
for the POI/event nearest-neighbor lookup.

```python
import pandas as pd
from fastmob.integration import join_with_pois, join_with_pois_by_category, join_with_events

traj = pd.DataFrame(...)   # lat, lng, datetime columns
pois = pd.DataFrame(...)   # lat, lng, id, name_poi, type_poi columns
events = pd.DataFrame(...) # lat, lng, datetime, event_id, event_type columns

with_pois = join_with_pois(traj, pois)
with_pois_by_cat = join_with_pois_by_category(traj, pois)
with_events = join_with_events(traj, events, time_window_s=900)
```

A trajectory point with no in-window event, or a `pois`/`events` frame with
zero rows, reports a real null (`None`/`inf`) rather than the untyped
`NaN`/`inf` sentinel PyMove itself uses.

---

::: fastmob.integration.join_with_pois
    options:
      show_source: false

---

::: fastmob.integration.join_with_pois_by_category
    options:
      show_source: false

---

::: fastmob.integration.join_with_events
    options:
      show_source: false
