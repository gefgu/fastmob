# Cleaning and segmentation

**Question.** Raw GPS traces contain glitches and recording gaps. What should I remove or split before measuring anything?

This page explains the ideas and the choices you must make. To run them on real data, open the worked notebook [Clean noisy GPS data and find stops](clean-and-detect-stops.md).

## Concepts

| Term | Meaning |
| --- | --- |
| Fix | One recorded position: person, time, latitude, longitude. |
| Outlier | A fix flagged as a recording error. The default `speed` filter flags a fix that would need an impossible speed to reach from its neighbours; the other filters use other criteria. |
| Observation gap | A pause in recording, for example a switched-off device. The person kept moving; the data did not. |
| Segment | A continuous stretch of recording. Each person's segment numbers restart at 0. |

## Choices that change your results

### Removing glitches: `filter`

| Method | What it flags | Main setting (default) |
| --- | --- | --- |
| `speed` (default) | a fix that implies a speed above the limit from its neighbours | `max_speed_kmh=500`; choose it from the fastest transport mode in your data (a walking-only study needs a far lower limit) |
| `hampel` | a speed that is far from the median speed of nearby points | `window_size=5`, `n_sigma=3` |
| `greedy` | a point inconsistent with the one before it at a given speed | `max_speed_kmh=100` |
| `smart_greedy` | the same test, keeping the longest run of mutually consistent points | `max_speed_kmh=100` |
| `zheng` | points outside runs of consistent points, with a minimum run length | `max_speed_kmh=100`, `min_seg_size=1` |

### Splitting traces: `segment`

`segment` never drops rows: it adds a `segment_id`, restarting at 0 for each person. The default method is `value_change`, which needs a column to watch (`col_name`); the others work from time and position alone.

| Method | A new segment starts when | Main setting (default) |
| --- | --- | --- |
| `observation_gap` | the time since the previous fix exceeds a gap | `gap_s=3600` |
| `stop` | the person stays within a radius for long enough | `stop_radius_km=0.2`, `minutes_for_a_stop=20` |
| `speed` | the person switches between moving and not moving | `speed_kmh=0`, `duration_s=300` |
| `temporal` | a calendar boundary is crossed | `mode="day"` |
| `angle_change` | the heading changes by more than an angle | `min_angle=45` |
| `value_change` (default) | the watched column changes | `col_name` |

### Thinning the data

| Function | What it does | Choose by |
| --- | --- | --- |
| `compress` | drops stationary clutter within a radius | whether you need time information afterwards; thinned data should not be used for waiting times |
| `simplify` | keeps the shape of the path (Douglas-Peucker) | the tolerance you can accept on the path |

The worked notebook runs the `speed` filter (with a 200 km/h limit), the `observation_gap` segmentation, and `compress` and `simplify`: see [Remove implausible jumps](clean-and-detect-stops.md#remove-implausible-jumps), [Split each trace at recording gaps](clean-and-detect-stops.md#split-each-trace-at-recording-gaps) and [Reduce the number of points](clean-and-detect-stops.md#reduce-the-number-of-points). [Fill gaps, smooth, compare and convert trajectories](process-trajectories.md) shows more segmentation.

## Assumptions to state in your methods

- Time order and time zone: fixes are ordered by `datetime` and used as given, with no time-zone conversion.
- Distances are straight lines on the sphere, in kilometres.
- Thresholds are not universal. Report the ones you used.

## Next steps

Run it: [Clean noisy GPS data and find stops](clean-and-detect-stops.md). Then detect stops: [Stops and locations](staypoints-and-locations.md).

## Related API

[`filter`](../reference/preprocessing.md#fastmob.preprocessing.filter) ·
[`segment`](../reference/preprocessing.md#fastmob.preprocessing.segment) ·
[`compress`](../reference/preprocessing.md#fastmob.preprocessing.compress) ·
[`simplify`](../reference/preprocessing.md#fastmob.preprocessing.simplify)
