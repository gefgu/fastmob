# Preprocessing

Clean, segment, thin and summarise trajectories, detect stops and places, and convert positions to H3 cells or origin-destination flows. Every function accepts any eager dataframe and returns the same backend. For a worked example on real GPS data, see [Clean noisy GPS data and find stops](../learn/clean-and-detect-stops.md); for the ideas behind the choices, see [Cleaning and segmentation](../learn/clean-and-segment.md).

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`filter`](#fastmob.preprocessing.filter) | Filter trajectory noise by removing outlier points using a named algorithm. | [Remove implausible jumps](../learn/clean-and-detect-stops.md#remove-implausible-jumps) |
| [`compress`](#fastmob.preprocessing.compress) | Compress trajectory by collapsing nearby points into single representative points. | [Reduce the number of points](../learn/clean-and-detect-stops.md#reduce-the-number-of-points) |
| [`stay_locations`](#fastmob.preprocessing.stay_locations) | Detect stay locations (stops) in trajectory data. | [Detect stops](../learn/clean-and-detect-stops.md#detect-stops) |
| [`cluster`](#fastmob.preprocessing.cluster) | Cluster stop locations with H3 cells and connected components. | [Group stops into locations](../learn/clean-and-detect-stops.md#group-stops-into-locations) |
| [`latlng_to_h3`](#fastmob.preprocessing.latlng_to_h3) | Convert each row's (lat, lng) to an H3 cell index, Rust-accelerated. |  |
| [`h3_to_latlng`](#fastmob.preprocessing.h3_to_latlng) | Convert each H3 cell to its geographic center coordinates. |  |
| [`trajectory_to_od`](#fastmob.preprocessing.trajectory_to_od) | Build a long-format Origin-Destination matrix directly from a raw trajectory. |  |
| [`trajectory_to_trips`](#fastmob.preprocessing.trajectory_to_trips) | Build a `fastmob.Trips` row for each consecutive ping pair. |  |
| [`segment`](#fastmob.preprocessing.segment) | Partition a trajectory into segments using a named algorithm. | [Split each trace at recording gaps](../learn/clean-and-detect-stops.md#split-each-trace-at-recording-gaps) |
| [`simplify`](#fastmob.preprocessing.simplify) | Simplify a trajectory by dropping points using a named algorithm. | [Reduce the number of points](../learn/clean-and-detect-stops.md#reduce-the-number-of-points) |
| [`identify_locations`](#fastmob.preprocessing.identify_locations) | Label each user's recurring `Locations` as `"home"`, `"work"`, or `"other"`. |  |
| [`predict_transport_mode`](#fastmob.preprocessing.predict_transport_mode) | Classify each tripleg's transport mode from its average speed. |  |
| [`calculate_modal_split`](#fastmob.preprocessing.calculate_modal_split) | Aggregate a `mode`-labeled Triplegs table into a modal-split table. |  |
| [`create_activity_flag`](#fastmob.preprocessing.create_activity_flag) | Flag each staypoint as a genuine "activity" (vs. a transient stop) by dwell time. |  |
| [`cdr_to_trips_df`](#fastmob.preprocessing.cdr_to_trips_df) | Derive a CDR trips dataframe from `cdr_to_visitation_df` output. |  |
| [`cdr_to_visitation_df`](#fastmob.preprocessing.cdr_to_visitation_df) | Convert raw CDR records into a NetMob-like visitation dataframe. |  |
<!-- api-table:end -->

::: fastmob.preprocessing.filter
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Remove implausible jumps](../learn/clean-and-detect-stops.md#remove-implausible-jumps)

---

::: fastmob.preprocessing.compress
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Reduce the number of points](../learn/clean-and-detect-stops.md#reduce-the-number-of-points)

---

::: fastmob.preprocessing.stay_locations
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Detect stops](../learn/clean-and-detect-stops.md#detect-stops)

---

::: fastmob.preprocessing.cluster
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Group stops into locations](../learn/clean-and-detect-stops.md#group-stops-into-locations)

---

::: fastmob.preprocessing.latlng_to_h3
    options:
      show_source: false

---

::: fastmob.preprocessing.h3_to_latlng
    options:
      show_source: false

---

::: fastmob.preprocessing.trajectory_to_od
    options:
      show_source: false

---

::: fastmob.preprocessing.trajectory_to_trips
    options:
      show_source: false

---

::: fastmob.preprocessing.segment
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Split each trace at recording gaps](../learn/clean-and-detect-stops.md#split-each-trace-at-recording-gaps)

---

::: fastmob.preprocessing.simplify
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Reduce the number of points](../learn/clean-and-detect-stops.md#reduce-the-number-of-points)

---

::: fastmob.preprocessing.identify_locations
    options:
      show_source: false

---

::: fastmob.preprocessing.predict_transport_mode
    options:
      show_source: false

---

::: fastmob.preprocessing.calculate_modal_split
    options:
      show_source: false

---

::: fastmob.preprocessing.create_activity_flag
    options:
      show_source: false

---

::: fastmob.preprocessing.cdr_to_trips_df
    options:
      show_source: false

---

::: fastmob.preprocessing.cdr_to_visitation_df
    options:
      show_source: false
