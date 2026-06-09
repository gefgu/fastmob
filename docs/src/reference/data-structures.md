# Data Structures

| API | Description |
| --- | --- |
| [`TrajDataFrame`](#skmob2.core.TrajDataFrame) | Narwhals-backed wrapper for trajectory data. |
| [`FlowDataFrame`](#skmob2.core.FlowDataFrame) | Narwhals-backed wrapper for origin-destination flow data. |

---

## `TrajDataFrame`

::: skmob2.core.TrajDataFrame
    options:
      show_source: false

### Measure methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.jump_lengths`](#skmob2.core.trajectory_dataframe.TrajDataFrame.jump_lengths) | Compute jump lengths (km) between consecutive GPS points. |
| [`TrajDataFrame.radius_of_gyration`](#skmob2.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration) | Compute the radius of gyration (km) for each user. |

::: skmob2.core.trajectory_dataframe.TrajDataFrame.jump_lengths
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.radius_of_gyration
    options:
      show_source: false

---

### Preprocessing methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.compress`](#skmob2.core.trajectory_dataframe.TrajDataFrame.compress) | Compress the trajectory by collapsing nearby consecutive points. |
| [`TrajDataFrame.stay_locations`](#skmob2.core.trajectory_dataframe.TrajDataFrame.stay_locations) | Detect stay locations (stops) in the trajectory. |

::: skmob2.core.trajectory_dataframe.TrajDataFrame.compress
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.stay_locations
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.to_flowdataframe`](#skmob2.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe) | Aggregate trajectory into a `FlowDataFrame` using a tessellation. Requires `skmob2[data]`. |
| [`TrajDataFrame.to_geodataframe`](#skmob2.core.trajectory_dataframe.TrajDataFrame.to_geodataframe) | Convert to a `geopandas.GeoDataFrame` with Point geometry. Requires `skmob2[data]`. |
| [`TrajDataFrame.mapping`](#skmob2.core.trajectory_dataframe.TrajDataFrame.mapping) | Assign each point to a tessellation tile. Requires `skmob2[data]`. |

::: skmob2.core.trajectory_dataframe.TrajDataFrame.to_flowdataframe
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.to_geodataframe
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.mapping
    options:
      show_source: false

---

### Utility methods

| API | Description |
| --- | --- |
| [`TrajDataFrame.sort_by_uid_and_datetime`](#skmob2.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime) | Return a copy sorted by user ID then datetime. |
| [`TrajDataFrame.settings_from`](#skmob2.core.trajectory_dataframe.TrajDataFrame.settings_from) | Copy metadata attributes from another `TrajDataFrame`. |
| [`TrajDataFrame.timezone_conversion`](#skmob2.core.trajectory_dataframe.TrajDataFrame.timezone_conversion) | Convert the datetime column from one timezone to another, in place. |

::: skmob2.core.trajectory_dataframe.TrajDataFrame.sort_by_uid_and_datetime
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.settings_from
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.timezone_conversion
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `visualization` extra:

```bash
pip install "skmob2[visualization]"
```

| API | Description |
| --- | --- |
| [`TrajDataFrame.plot_trajectory`](#skmob2.core.trajectory_dataframe.TrajDataFrame.plot_trajectory) | Plot trajectories on an interactive Folium map. |
| [`TrajDataFrame.plot_stops`](#skmob2.core.trajectory_dataframe.TrajDataFrame.plot_stops) | Plot detected stop locations on an interactive Folium map. |
| [`TrajDataFrame.plot_diary`](#skmob2.core.trajectory_dataframe.TrajDataFrame.plot_diary) | Plot a mobility diary as a coloured time-span chart. |

::: skmob2.core.trajectory_dataframe.TrajDataFrame.plot_trajectory
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.plot_stops
    options:
      show_source: false

---

::: skmob2.core.trajectory_dataframe.TrajDataFrame.plot_diary
    options:
      show_source: false

---

## `FlowDataFrame`

::: skmob2.core.FlowDataFrame
    options:
      show_source: false

### Query methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.get_flow`](#skmob2.core.flow_dataframe.FlowDataFrame.get_flow) | Return the flow between two tile IDs (0 if absent). |
| [`FlowDataFrame.get_geometry`](#skmob2.core.flow_dataframe.FlowDataFrame.get_geometry) | Return the geometry of a tessellation tile. |
| [`FlowDataFrame.settings_from`](#skmob2.core.flow_dataframe.FlowDataFrame.settings_from) | Copy metadata attributes from another `FlowDataFrame`. |

::: skmob2.core.flow_dataframe.FlowDataFrame.get_flow
    options:
      show_source: false

---

::: skmob2.core.flow_dataframe.FlowDataFrame.get_geometry
    options:
      show_source: false

---

::: skmob2.core.flow_dataframe.FlowDataFrame.settings_from
    options:
      show_source: false

---

### Conversion methods

| API | Description |
| --- | --- |
| [`FlowDataFrame.to_matrix`](#skmob2.core.flow_dataframe.FlowDataFrame.to_matrix) | Convert the flow table to a square numpy matrix. |

::: skmob2.core.flow_dataframe.FlowDataFrame.to_matrix
    options:
      show_source: false

---

### Visualization methods

Visualization methods require the optional `visualization` extra:

```bash
pip install "skmob2[visualization]"
```

| API | Description |
| --- | --- |
| [`FlowDataFrame.plot_flows`](#skmob2.core.flow_dataframe.FlowDataFrame.plot_flows) | Plot origin-destination flows on an interactive Folium map. |
| [`FlowDataFrame.plot_tessellation`](#skmob2.core.flow_dataframe.FlowDataFrame.plot_tessellation) | Plot the spatial tessellation on an interactive Folium map. |

::: skmob2.core.flow_dataframe.FlowDataFrame.plot_flows
    options:
      show_source: false

---

::: skmob2.core.flow_dataframe.FlowDataFrame.plot_tessellation
    options:
      show_source: false
