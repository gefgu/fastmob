# Cleaning and segmentation

**Question.** Raw GPS traces contain glitches and recording gaps. What should I remove or split before measuring anything?

This page explains the ideas and the choices you must make. To run them on real data, open the worked notebook [Clean noisy GPS data and find stops](clean-and-detect-stops.md).

## Concepts

| Term | Meaning |
| --- | --- |
| Fix | One recorded position: person, time, latitude, longitude. |
| Outlier | A fix that would need an impossible speed to reach from its neighbours, usually a receiver glitch. |
| Observation gap | A pause in recording, for example a switched-off device. The person kept moving; the data did not. |
| Segment | A continuous stretch of recording. Each person's segment numbers restart at 0. |

## Choices that change your results

| Decision | Options in Fastkit-Mobility | Choose by |
| --- | --- | --- |
| Remove glitches | `filter` with `method="speed"` (default), `hampel`, `greedy`, `smart_greedy`, `zheng` | the fastest transport mode in your data; a walking-only study needs a far lower speed limit than a mixed one |
| Split traces | `segment` with `observation_gap`, `stop`, `speed`, `temporal`, `angle_change`, `value_change` | what a segment must mean: a continuous recording, a trip, or a time window |
| Thin the data | `compress` (drops stationary clutter), `simplify` (keeps path shape) | whether you need time information afterwards; thinned data should not be used for waiting times |

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
