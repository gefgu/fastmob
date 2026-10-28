# Stops and locations

**Question.** Where did each person actually stay, which places do they return to, and what are those places for?

This page explains the concepts and the typed hierarchy built on them. To detect stops on real data, open [Clean noisy GPS data and find stops](clean-and-detect-stops.md).

## Concepts

| Term | Meaning |
| --- | --- |
| Staypoint | A stop: at least `minutes_for_a_stop` minutes within `spatial_radius_km` of the arrival point. |
| Location | A place a person returns to, found by clustering their staypoints. Location ids are per person. |
| Activity | A staypoint long enough to count as an activity, as opposed to a pause during travel (`create_activity_flag`, default 15 minutes). |

## The hierarchy

Position fixes become staypoints; the movement between staypoints becomes triplegs; triplegs join into trips; trips that return to the same location form tours. Each level is a typed dataframe (`Positionfixes`, `Staypoints`, `Triplegs`, `Trips`, `Tours`). See [data structures](../reference/data-structures.md) for each level's columns.

```python
from fastmob import Positionfixes

fixes = Positionfixes(traj)
staypoints = fixes.generate_staypoints(minutes_for_a_stop=20, spatial_radius_km=0.2)
locations, assigned_staypoints = staypoints.generate_user_locations(epsilon_km=0.2)
labeled_locations = locations.identify(assigned_staypoints)
```

`identify` labels, for each person, the location with the most **night-time dwell** (22:00 to 06:00) as `home`, and the other location with the most **weekday daytime dwell** (Monday to Friday, 08:00 to 18:00) as `work`; every other location is `other`. These are hours in the data's own time zone, and they count time spent, not number of visits. [`home_location` and `work_location`](individual-and-collective-measures.md) use check-in counts instead, so on the same data they can disagree. Treat the labels as hypotheses and check them against local time and your own knowledge of the data.

To run this code on real GPS traces, see [Build the trajectory hierarchy](trajectory-hierarchy.md); it applies the same calls to GeoLife and shows each table.

## Choices that change your results

- **Duration and radius:** shorter durations find more, shorter stops; a larger radius merges nearby places.
- **Silence limit** (`no_data_for_minutes`): without it, a device that goes quiet at a place can produce a stay lasting days. With it, a stay in progress when the data go silent for longer than the limit is discarded, not shortened, so you also lose real stays that coincide with recording gaps.
- **Cluster scale** (`epsilon_km` in `generate_user_locations`, `cluster_radius_km` in `cluster`; the same knob in the typed and the functional API): `cluster` places stops on an H3 hexagonal grid and joins occupied cells that touch, so the radius only picks the grid size. Places are approximate, and a chain of touching cells becomes one location. Pass `h3_resolution` to choose the grid explicitly.

## Related API

[`stay_locations`](../reference/preprocessing.md#fastmob.preprocessing.stay_locations) ·
[`cluster`](../reference/preprocessing.md#fastmob.preprocessing.cluster) ·
[`identify_locations`](../reference/preprocessing.md#fastmob.preprocessing.identify_locations) ·
[`create_activity_flag`](../reference/preprocessing.md#fastmob.preprocessing.create_activity_flag) ·
[`Positionfixes`](../reference/data-structures/positionfixes.md) ·
[`Staypoints`](../reference/data-structures/staypoints.md) (`generate_staypoints`, `generate_user_locations`) ·
[`Locations`](../reference/data-structures/locations.md) (`identify`) ·
[`Triplegs`](../reference/data-structures/triplegs.md) ·
[`Trips`](../reference/data-structures/trips.md) ·
[`Tours`](../reference/data-structures/tours.md)
