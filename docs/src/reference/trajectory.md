# Trajectory

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`interpolate`](#fastmob.trajectory.interpolate) | Fill gaps in a trajectory by inserting interpolated points. | [Fill recording gaps](../learn/process-trajectories.md#fill-recording-gaps) |
| [`interpolate_at`](#fastmob.trajectory.interpolate_at) | Query each user's interpolated position at one or more timestamps. | [Ask for a position at a given time](../learn/process-trajectories.md#ask-for-a-position-at-a-given-time) |
| [`trajectory_distance`](#fastmob.trajectory.trajectory_distance) | Compute a similarity/distance metric between two trajectories. | [Compare two trajectories](../learn/process-trajectories.md#compare-two-trajectories) |
| [`smooth`](#fastmob.trajectory.smooth) | Smooth a trajectory's positions using a named algorithm. | [Smooth GPS noise](../learn/process-trajectories.md#smooth-gps-noise) |
| [`cluster_trajectory_shapes`](#fastmob.trajectory.cluster_trajectory_shapes) | Cluster a list of independent trajectories by their overall shape. | [Group trajectories by shape](../learn/process-trajectories.md#group-trajectories-by-shape) |
| [`cluster_trajectory_shapes_from_segments`](#fastmob.trajectory.cluster_trajectory_shapes_from_segments) | Cluster trajectory shapes directly from a `segment()`-output (or `Triplegs.points`) frame, splitting into per-segment trajectories internally. | [Group trajectories by shape](../learn/process-trajectories.md#group-trajectories-by-shape) |
<!-- api-table:end -->

::: fastmob.trajectory.interpolate
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Fill recording gaps](../learn/process-trajectories.md#fill-recording-gaps)

---

::: fastmob.trajectory.interpolate_at
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Ask for a position at a given time](../learn/process-trajectories.md#ask-for-a-position-at-a-given-time)

---

::: fastmob.trajectory.trajectory_distance
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Compare two trajectories](../learn/process-trajectories.md#compare-two-trajectories)

---

::: fastmob.trajectory.smooth
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Smooth GPS noise](../learn/process-trajectories.md#smooth-gps-noise)

---

::: fastmob.trajectory.cluster_trajectory_shapes
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Group trajectories by shape](../learn/process-trajectories.md#group-trajectories-by-shape)

---

::: fastmob.trajectory.cluster_trajectory_shapes_from_segments
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Group trajectories by shape](../learn/process-trajectories.md#group-trajectories-by-shape)

