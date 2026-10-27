---
icon: lucide/route
---

# TrajDataFrame

`TrajDataFrame` is fastmob's general-purpose wrapper for timestamped latitude
and longitude observations. It accepts eager Narwhals-compatible DataFrames,
detects conventional column names, and records the resolved user, time, and
coordinate metadata for later operations.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`TrajDataFrame`](#fastmob.core.trajectory_dataframe.TrajDataFrame) | Narwhals-backed wrapper for trajectory data. |  |
| [`compress`](#fastmob.core.trajectory_dataframe.TrajDataFrame.compress) | Compress the trajectory by collapsing nearby consecutive points. | [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) |
| [`interpolate`](#fastmob.core.trajectory_dataframe.TrajDataFrame.interpolate) | Fill gaps in the trajectory using a named interpolation algorithm. | [Fill recording gaps](../../learn/process-trajectories.md#fill-recording-gaps) |
| [`interpolate_at`](#fastmob.core.trajectory_dataframe.TrajDataFrame.interpolate_at) | Query the interpolated position of each user at one or more timestamps. | [Ask for a position at a given time](../../learn/process-trajectories.md#ask-for-a-position-at-a-given-time) |
| [`jump_lengths`](#fastmob.core.trajectory_dataframe.TrajDataFrame.jump_lengths) | Compute the jump lengths (km) between consecutive GPS points. | [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) |
| [`mapping`](#fastmob.core.trajectory_dataframe.TrajDataFrame.mapping) | Assign each trajectory point to a tile in a spatial tessellation. | [Observed flows between zones](../../learn/fit-and-evaluate.md#observed-flows-between-zones) |
| [`radius_of_gyration`](#fastmob.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration) | Compute the radius of gyration (km) for each user. | [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) |
| [`settings_from`](#fastmob.core.trajectory_dataframe.TrajDataFrame.settings_from) | Copy metadata attributes from another TrajDataFrame. | [Move between tables](../../learn/process-trajectories.md#move-between-tables) |
| [`smooth`](#fastmob.core.trajectory_dataframe.TrajDataFrame.smooth) | Smooth the trajectory's positions using a named algorithm. | [Smooth GPS noise](../../learn/process-trajectories.md#smooth-gps-noise) |
| [`sort_by_uid_and_datetime`](#fastmob.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime) | Return a copy sorted by user ID then datetime. | [Keep the data in time order](../../learn/process-trajectories.md#keep-the-data-in-time-order) |
| [`stay_locations`](#fastmob.core.trajectory_dataframe.TrajDataFrame.stay_locations) | Detect stay locations (stops) in the trajectory. | [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) |
| [`timezone_conversion`](#fastmob.core.trajectory_dataframe.TrajDataFrame.timezone_conversion) | Convert the datetime column from one timezone to another, in place. | [Build the visits](../../learn/motifs.md#build-the-visits) |
| [`to_flowdataframe`](#fastmob.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe) | Aggregate the trajectory into a FlowDataFrame using a tessellation. | [Observed flows between zones](../../learn/fit-and-evaluate.md#observed-flows-between-zones) |
| [`to_geodataframe`](#fastmob.core.trajectory_dataframe.TrajDataFrame.to_geodataframe) | Convert to a `geopandas.GeoDataFrame` with Point geometry. | [Move between tables](../../learn/process-trajectories.md#move-between-tables) |
| [`trajectory_distance`](#fastmob.core.trajectory_dataframe.TrajDataFrame.trajectory_distance) | Compute a similarity/distance metric against another trajectory's point-sequence. | [Compare two trajectories](../../learn/process-trajectories.md#compare-two-trajectories) |
| [`work_location`](#fastmob.core.trajectory_dataframe.TrajDataFrame.work_location) | Infer the most-visited weekday daytime location for each user. | [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) |
<!-- api-table:end -->

## Required data

Provide latitude, longitude, and datetime columns; a user-ID column is optional.
Pass explicit column names when automatic detection is not appropriate. The
wrapper preserves the native backend and can sort observations by user and time.

```python
import pandas as pd
from fastmob import TrajDataFrame

traj = TrajDataFrame(pd.DataFrame({
    "uid": [1, 1],
    "datetime": ["2024-01-01 08:00", "2024-01-01 08:10"],
    "lat": [47.37, 47.38],
    "lng": [8.54, 8.55],
}), sort=True)

jumps = traj.jump_lengths()
```

For typed hierarchy generation, use [`Positionfixes`](positionfixes.md), which
extends this class.

## API

::: fastmob.core.trajectory_dataframe.TrajDataFrame
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Summaries from the same object](../../learn/process-trajectories.md#summaries-from-the-same-object) · [Fill recording gaps](../../learn/process-trajectories.md#fill-recording-gaps) · [Ask for a position at a given time](../../learn/process-trajectories.md#ask-for-a-position-at-a-given-time) · [Observed flows between zones](../../learn/fit-and-evaluate.md#observed-flows-between-zones) · [Move between tables](../../learn/process-trajectories.md#move-between-tables) · [Smooth GPS noise](../../learn/process-trajectories.md#smooth-gps-noise) · [Keep the data in time order](../../learn/process-trajectories.md#keep-the-data-in-time-order) · [Build the visits](../../learn/motifs.md#build-the-visits) · [Compare two trajectories](../../learn/process-trajectories.md#compare-two-trajectories)

