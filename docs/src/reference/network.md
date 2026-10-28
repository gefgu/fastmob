# Network

Road/rail-network-constrained distance: build a routable graph from
Overture Maps transportation data, snap trajectory points to it, and query
network (not straight-line) distance via a Rust contraction-hierarchy router.

<!-- api-table:start -->
**Build a network**

| API | Description | Example |
| --- | --- | --- |
| [`fetch_road_network`](#fastmob.network.fetch_road_network) | Fetch and build a car-routable graph from Overture road segments. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |
| [`build_road_graph`](#fastmob.network.build_road_graph) | Load a saved road graph from disk, or fetch it from Overture Maps and save it. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |
| [`fetch_rail_network`](#fastmob.network.fetch_rail_network) | Fetch rail segments from Overture Maps and build a bidirectional rail graph. | [Rail network](../learn/road-network-analysis.md#rail-network) |
| [`build_rail_graph`](#fastmob.network.build_rail_graph) | Load a saved rail graph from disk, or fetch it from Overture Maps and save it. | [Rail network](../learn/road-network-analysis.md#rail-network) |
| [`RoadNetwork`](#fastmob.network.RoadNetwork) | A road (or rail) network prepared once (contraction hierarchy) and reused for many point-to-point physical-distance queries. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.batch_distances`](#fastmob.network.RoadNetwork.batch_distances) | Batch route-length (metres) query for `(from_node, to_node)` pairs. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.batch_routes`](#fastmob.network.RoadNetwork.batch_routes) | Batch route-geometry query for `(from_node, to_node)` pairs. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.build`](#fastmob.network.RoadNetwork.build) | Prepare a contraction hierarchy from a road/rail graph's edges. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |

**Snap, route and match**

| API | Description | Example |
| --- | --- | --- |
| [`snap_locations_to_graph`](#fastmob.network.snap_locations_to_graph) | Snap each tessellation row to its nearest road/rail graph node. | [Snap points to the network](../learn/road-network-analysis.md#snap-points-to-the-network) |
| [`haversine_m_batch`](#fastmob.network.haversine_m_batch) | Elementwise Haversine distance (metres) between two same-length arrays of points, computed in parallel. |  |
| [`match_trajectory`](#fastmob.network.match_trajectory) | Map-match a trajectory with a directed-edge HMM/Viterbi model. | [Match a GPS trace to roads](../learn/road-network-analysis.md#match-a-gps-trace-to-roads) |
| [`od_desire_lines`](#fastmob.network.od_desire_lines) | Aggregate OD-pair flows onto the road/rail graph's edges (desire lines). | [Accumulate flows on roads](../learn/road-network-analysis.md#accumulate-flows-on-roads) |

**Measures along the road network**

| API | Description | Example |
| --- | --- | --- |
| [`jump_lengths_road`](#fastmob.measures.individual.network_distance.jump_lengths_road) | Jump lengths (km) measured along the road network. | [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line) |
| [`radius_of_gyration_road`](#fastmob.measures.individual.network_distance.radius_of_gyration_road) | Radius of gyration (km) per user, measured along the road network. | [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line) |
<!-- api-table:end -->

`fetch_road_network` and `fetch_rail_network` require DuckDB: install it with
`pip install duckdb`. Snapping uses Fastkit-Mobility's native Rust spatial index and
has no additional Python dependency.

For a worked walk-through (build a network, snap points, compare road and
straight-line distances, accumulate flows, match a GPS trace) see
[Road networks: matching, routing and route flows](../learn/road-network-analysis.md) and the concept
page [Roads, rails and map matching](../learn/network-analysis.md). Node ids passed to the
distance, route and desire-line functions come from `snap_locations_to_graph`;
an unsnapped node is `-1`.

---

::: fastmob.network.fetch_road_network
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network)

---

::: fastmob.network.build_road_graph
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network)

---

::: fastmob.network.fetch_rail_network
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Rail network](../learn/road-network-analysis.md#rail-network)

---

::: fastmob.network.build_rail_graph
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Rail network](../learn/road-network-analysis.md#rail-network)

---

::: fastmob.network.RoadNetwork
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) · [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network)

---

::: fastmob.network.snap_locations_to_graph
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Snap points to the network](../learn/road-network-analysis.md#snap-points-to-the-network)

---

::: fastmob.network.haversine_m_batch
    options:
      show_source: false

---

::: fastmob.network.match_trajectory
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Match a GPS trace to roads](../learn/road-network-analysis.md#match-a-gps-trace-to-roads)

---

::: fastmob.network.od_desire_lines
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Accumulate flows on roads](../learn/road-network-analysis.md#accumulate-flows-on-roads)

---

::: fastmob.measures.individual.network_distance.jump_lengths_road
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line)

---

::: fastmob.measures.individual.network_distance.radius_of_gyration_road
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line)

