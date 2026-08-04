# Data Structures

| API | Description |
| --- | --- |
| [`TrajDataFrame`](#fastmob.core.TrajDataFrame) | Narwhals-backed wrapper for trajectory data. |
| [`FlowDataFrame`](#fastmob.core.FlowDataFrame) | Narwhals-backed wrapper for origin-destination flow data. |

---

## `TrajDataFrame`

::: fastmob.core.TrajDataFrame
    options:
      show_source: false

### Measure methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.jump_lengths`](#fastmob.core.trajectory_dataframe.TrajDataFrame.jump_lengths) | Compute jump lengths (km) between consecutive GPS points. |
| [`TrajDataFrame.radius_of_gyration`](#fastmob.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration) | Compute the radius of gyration (km) for each user. |

::: fastmob.core.trajectory_dataframe.TrajDataFrame.jump_lengths
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration
    options:
      show_source: false

---

### Preprocessing methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.compress`](#fastmob.core.trajectory_dataframe.TrajDataFrame.compress) | Compress the trajectory by collapsing nearby consecutive points. |
| [`TrajDataFrame.stay_locations`](#fastmob.core.trajectory_dataframe.TrajDataFrame.stay_locations) | Detect stay locations (stops) in the trajectory. |

::: fastmob.core.trajectory_dataframe.TrajDataFrame.compress
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.stay_locations
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.to_flowdataframe`](#fastmob.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe) | Aggregate trajectory into a `FlowDataFrame` using a tessellation. Requires `fastmob[data]`. |
| [`TrajDataFrame.to_geodataframe`](#fastmob.core.trajectory_dataframe.TrajDataFrame.to_geodataframe) | Convert to a `geopandas.GeoDataFrame` with Point geometry. Requires `fastmob[data]`. |
| [`TrajDataFrame.mapping`](#fastmob.core.trajectory_dataframe.TrajDataFrame.mapping) | Assign each point to a tessellation tile. Requires `fastmob[data]`. |

::: fastmob.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.to_geodataframe
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.mapping
    options:
      show_source: false

---

### Comparison methods

`compare_to` is defined once on `BaseDataFrame` and inherited by every hierarchy
class (`TrajDataFrame`, `FlowDataFrame`, `Staypoints`, `Trips`, `Triplegs`,
`Locations`, `Tours`).

| API | Description |
| --- | --- |
| [`TrajDataFrame.compare_to`](#fastmob.core.base.BaseDataFrame.compare_to) | Compare a value column against another instance, optionally grouped by an existing column. |

::: fastmob.core.base.BaseDataFrame.compare_to
    options:
      show_source: false

---

### Utility methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.sort_by_uid_and_datetime`](#fastmob.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime) | Return a copy sorted by user ID then datetime. |
| [`TrajDataFrame.settings_from`](#fastmob.core.trajectory_dataframe.TrajDataFrame.settings_from) | Copy metadata attributes from another `TrajDataFrame`. |
| [`TrajDataFrame.timezone_conversion`](#fastmob.core.trajectory_dataframe.TrajDataFrame.timezone_conversion) | Convert the datetime column from one timezone to another, in place. |

::: fastmob.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.settings_from
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.timezone_conversion
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `vis` extra:

```bash
pip install "fastmob[vis]"
```

| API | Description |
| --- | --- |
| [`TrajDataFrame.plot_trajectory`](#fastmob.core.trajectory_dataframe.TrajDataFrame.plot_trajectory) | Plot trajectories on an interactive Folium map. |
| [`TrajDataFrame.plot_stops`](#fastmob.core.trajectory_dataframe.TrajDataFrame.plot_stops) | Plot detected stop locations on an interactive Folium map. |
| [`TrajDataFrame.plot_diary`](#fastmob.core.trajectory_dataframe.TrajDataFrame.plot_diary) | Plot a mobility diary as a coloured time-span chart. |

::: fastmob.core.trajectory_dataframe.TrajDataFrame.plot_trajectory
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.plot_stops
    options:
      show_source: false

---

::: fastmob.core.trajectory_dataframe.TrajDataFrame.plot_diary
    options:
      show_source: false

---

## `FlowDataFrame`

::: fastmob.core.FlowDataFrame
    options:
      show_source: false

### Query methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.get_flow`](#fastmob.core.flow_dataframe.FlowDataFrame.get_flow) | Return the flow between two tile IDs (0 if absent). |
| [`FlowDataFrame.get_geometry`](#fastmob.core.flow_dataframe.FlowDataFrame.get_geometry) | Return the geometry of a tessellation tile. |
| [`FlowDataFrame.settings_from`](#fastmob.core.flow_dataframe.FlowDataFrame.settings_from) | Copy metadata attributes from another `FlowDataFrame`. |
| [`FlowDataFrame.compare_to`](#fastmob.core.base.BaseDataFrame.compare_to) | Compare a value column against another instance, optionally grouped by an existing column (inherited from `BaseDataFrame`, documented above). |

::: fastmob.core.flow_dataframe.FlowDataFrame.get_flow
    options:
      show_source: false

---

::: fastmob.core.flow_dataframe.FlowDataFrame.get_geometry
    options:
      show_source: false

---

::: fastmob.core.flow_dataframe.FlowDataFrame.settings_from
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.to_matrix`](#fastmob.core.flow_dataframe.FlowDataFrame.to_matrix) | Convert the flow table to a square numpy matrix. |

::: fastmob.core.flow_dataframe.FlowDataFrame.to_matrix
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `vis` extra:

```bash
pip install "fastmob[vis]"
```

| API | Description |
| --- | --- |
| [`FlowDataFrame.plot_flows`](#fastmob.core.flow_dataframe.FlowDataFrame.plot_flows) | Plot origin-destination flows on an interactive Folium map. |
| [`FlowDataFrame.plot_tessellation`](#fastmob.core.flow_dataframe.FlowDataFrame.plot_tessellation) | Plot the spatial tessellation on an interactive Folium map. |

::: fastmob.core.flow_dataframe.FlowDataFrame.plot_flows
    options:
      show_source: false

---

::: fastmob.core.flow_dataframe.FlowDataFrame.plot_tessellation
    options:
      show_source: false
