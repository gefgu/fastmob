# Data Structures

| API | Description |
| --- | --- |
| [`TrajDataFrame`](#fkmob.core.TrajDataFrame) | Narwhals-backed wrapper for trajectory data. |
| [`FlowDataFrame`](#fkmob.core.FlowDataFrame) | Narwhals-backed wrapper for origin-destination flow data. |

---

## `TrajDataFrame`

::: fkmob.core.TrajDataFrame
    options:
      show_source: false

### Measure methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.jump_lengths`](#fkmob.core.trajectory_dataframe.TrajDataFrame.jump_lengths) | Compute jump lengths (km) between consecutive GPS points. |
| [`TrajDataFrame.radius_of_gyration`](#fkmob.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration) | Compute the radius of gyration (km) for each user. |

::: fkmob.core.trajectory_dataframe.TrajDataFrame.jump_lengths
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration
    options:
      show_source: false

---

### Preprocessing methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.compress`](#fkmob.core.trajectory_dataframe.TrajDataFrame.compress) | Compress the trajectory by collapsing nearby consecutive points. |
| [`TrajDataFrame.stay_locations`](#fkmob.core.trajectory_dataframe.TrajDataFrame.stay_locations) | Detect stay locations (stops) in the trajectory. |

::: fkmob.core.trajectory_dataframe.TrajDataFrame.compress
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.stay_locations
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.to_flowdataframe`](#fkmob.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe) | Aggregate trajectory into a `FlowDataFrame` using a tessellation. Requires `fkmob[data]`. |
| [`TrajDataFrame.to_geodataframe`](#fkmob.core.trajectory_dataframe.TrajDataFrame.to_geodataframe) | Convert to a `geopandas.GeoDataFrame` with Point geometry. Requires `fkmob[data]`. |
| [`TrajDataFrame.mapping`](#fkmob.core.trajectory_dataframe.TrajDataFrame.mapping) | Assign each point to a tessellation tile. Requires `fkmob[data]`. |

::: fkmob.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.to_geodataframe
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.mapping
    options:
      show_source: false

---

### Utility methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.sort_by_uid_and_datetime`](#fkmob.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime) | Return a copy sorted by user ID then datetime. |
| [`TrajDataFrame.settings_from`](#fkmob.core.trajectory_dataframe.TrajDataFrame.settings_from) | Copy metadata attributes from another `TrajDataFrame`. |
| [`TrajDataFrame.timezone_conversion`](#fkmob.core.trajectory_dataframe.TrajDataFrame.timezone_conversion) | Convert the datetime column from one timezone to another, in place. |

::: fkmob.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.settings_from
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.timezone_conversion
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `visualization` extra:

```bash
pip install "fkmob[visualization]"
```

| API | Description |
| --- | --- |
| [`TrajDataFrame.plot_trajectory`](#fkmob.core.trajectory_dataframe.TrajDataFrame.plot_trajectory) | Plot trajectories on an interactive Folium map. |
| [`TrajDataFrame.plot_stops`](#fkmob.core.trajectory_dataframe.TrajDataFrame.plot_stops) | Plot detected stop locations on an interactive Folium map. |
| [`TrajDataFrame.plot_diary`](#fkmob.core.trajectory_dataframe.TrajDataFrame.plot_diary) | Plot a mobility diary as a coloured time-span chart. |

::: fkmob.core.trajectory_dataframe.TrajDataFrame.plot_trajectory
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.plot_stops
    options:
      show_source: false

---

::: fkmob.core.trajectory_dataframe.TrajDataFrame.plot_diary
    options:
      show_source: false

---

## `FlowDataFrame`

::: fkmob.core.FlowDataFrame
    options:
      show_source: false

### Query methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.get_flow`](#fkmob.core.flow_dataframe.FlowDataFrame.get_flow) | Return the flow between two tile IDs (0 if absent). |
| [`FlowDataFrame.get_geometry`](#fkmob.core.flow_dataframe.FlowDataFrame.get_geometry) | Return the geometry of a tessellation tile. |
| [`FlowDataFrame.settings_from`](#fkmob.core.flow_dataframe.FlowDataFrame.settings_from) | Copy metadata attributes from another `FlowDataFrame`. |

::: fkmob.core.flow_dataframe.FlowDataFrame.get_flow
    options:
      show_source: false

---

::: fkmob.core.flow_dataframe.FlowDataFrame.get_geometry
    options:
      show_source: false

---

::: fkmob.core.flow_dataframe.FlowDataFrame.settings_from
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.to_matrix`](#fkmob.core.flow_dataframe.FlowDataFrame.to_matrix) | Convert the flow table to a square numpy matrix. |

::: fkmob.core.flow_dataframe.FlowDataFrame.to_matrix
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `visualization` extra:

```bash
pip install "fkmob[visualization]"
```

| API | Description |
| --- | --- |
| [`FlowDataFrame.plot_flows`](#fkmob.core.flow_dataframe.FlowDataFrame.plot_flows) | Plot origin-destination flows on an interactive Folium map. |
| [`FlowDataFrame.plot_tessellation`](#fkmob.core.flow_dataframe.FlowDataFrame.plot_tessellation) | Plot the spatial tessellation on an interactive Folium map. |

::: fkmob.core.flow_dataframe.FlowDataFrame.plot_flows
    options:
      show_source: false

---

::: fkmob.core.flow_dataframe.FlowDataFrame.plot_tessellation
    options:
      show_source: false
