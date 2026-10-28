# Network

Road/rail-network-constrained distance: build a routable graph from
Overture Maps transportation data, snap trajectory points to it, and query
network (not straight-line) distance via a Rust contraction-hierarchy router.

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`RoadNetwork`](#fastmob.network.RoadNetwork) | A road (or rail) network prepared once (contraction hierarchy) and reused for many point-to-point physical-distance queries. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.batch_distances`](#fastmob.network.RoadNetwork.batch_distances) | Batch route-length (metres) query for `(from_node, to_node)` pairs. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.batch_routes`](#fastmob.network.RoadNetwork.batch_routes) | Batch route-geometry query for `(from_node, to_node)` pairs. | [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) |
| [`RoadNetwork.build`](#fastmob.network.RoadNetwork.build) | Prepare a contraction hierarchy from a road/rail graph's edges. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |
| [`fetch_road_network`](#fastmob.network.fetch_road_network) | Fetch and build a car-routable graph from Overture road segments. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |
| [`build_road_graph`](#fastmob.network.build_road_graph) | Load a saved road graph from disk, or fetch it from Overture Maps and save it. | [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network) |
| [`fetch_rail_network`](#fastmob.network.fetch_rail_network) | Fetch rail segments from Overture Maps and build a bidirectional rail graph. | [Rail network](../learn/road-network-analysis.md#rail-network) |
| [`build_rail_graph`](#fastmob.network.build_rail_graph) | Load a saved rail graph from disk, or fetch it from Overture Maps and save it. | [Rail network](../learn/road-network-analysis.md#rail-network) |
| [`snap_locations_to_graph`](#fastmob.network.snap_locations_to_graph) | Snap each tessellation row to its nearest road/rail graph node. | [Snap points to the network](../learn/road-network-analysis.md#snap-points-to-the-network) |
| [`haversine_m_batch`](#fastmob.network.haversine_m_batch) | Elementwise Haversine distance (metres) between two same-length arrays of points, computed in parallel. |  |
| [`od_desire_lines`](#fastmob.network.od_desire_lines) | Aggregate OD-pair flows onto the road/rail graph's edges (desire lines). | [Accumulate flows on roads](../learn/road-network-analysis.md#accumulate-flows-on-roads) |
| [`jump_lengths_road`](#fastmob.measures.individual.network_distance.jump_lengths_road) | Jump lengths (km) measured along the road network. | [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line) |
| [`radius_of_gyration_road`](#fastmob.measures.individual.network_distance.radius_of_gyration_road) | Radius of gyration (km) per user, measured along the road network. | [Road distance versus straight line](../learn/road-network-analysis.md#road-distance-versus-straight-line) |
| [`match_trajectory`](#fastmob.network.match_trajectory) | Map-match a trajectory with a directed-edge HMM/Viterbi model. | [Match a GPS trace to roads](../learn/road-network-analysis.md#match-a-gps-trace-to-roads) |
<!-- api-table:end -->

`fetch_road_network` and `fetch_rail_network` require DuckDB: install it with
`pip install duckdb`. Snapping uses Fastkit-Mobility's native Rust spatial index and
has no additional Python dependency.

## Network-aware distance measures

`fastmob.measures.individual.jump_lengths_road`/`radius_of_gyration_road` mirror
`jump_lengths`/`radius_of_gyration` but measure distance along a prepared
`RoadNetwork` instead of straight-line, falling back to Haversine per-pair
wherever a point is unsnapped or the graph is disconnected between the two
points:

```python
import pandas as pd
from fastmob.network import RoadNetwork, fetch_road_network
from fastmob.measures.individual import jump_lengths_road, radius_of_gyration_road

nodes_df, edges_df = fetch_road_network(2.34, 48.85, 2.36, 48.86, "2026-05-20.0")
network = RoadNetwork.build(edges_df, nodes_df)

traj = pd.DataFrame(...)  # uid, datetime, lat, lng columns
jumps_km = jump_lengths_road(traj, network=network)
rg_km = radius_of_gyration_road(traj, network=network)
```

## Route geometry and OD desire lines

`RoadNetwork.batch_routes` returns the actual waypoint path (not just total
distance) for a batch of `(from_node, to_node)` queries, decimated to at
most `max_waypoints` points per route (always keeping the first and last).
`od_desire_lines` is the Overture-native analogue of stplanr's
`overline`/`overline2`: it aggregates many origin-destination flows onto
the road/rail graph's edges, so overlapping trips accumulate onto shared
segments instead of remaining one separate desire line per pair.

```python
import numpy as np
from fastmob.network import RoadNetwork, od_desire_lines

network = RoadNetwork.build(edges_df, nodes_df)

routes = network.batch_routes(np.array([0, 5]), np.array([12, 3]), max_waypoints=50)
# columns: query_id, lat, lng, cum_weight_ds

edges_with_flow, dropped_flow = od_desire_lines(
    network, np.array([0, 1]), np.array([12, 12]), np.array([5.0, 3.0])
)
# columns: edge_from, edge_to, from_lat, from_lng, to_lat, to_lng, total_flow
```

Both fall back gracefully for unsnapped (negative node id) or
graph-disconnected queries: `batch_routes` contributes zero rows for that
query, and `od_desire_lines` adds that query's flow to `dropped_flow`
instead of any edge.

---

::: fastmob.network.RoadNetwork
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Distances and routes on the network](../learn/road-network-analysis.md#distances-and-routes-on-the-network) · [Trajectory and road network](../learn/road-network-analysis.md#trajectory-and-road-network)

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

::: fastmob.network.snap_locations_to_graph
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Snap points to the network](../learn/road-network-analysis.md#snap-points-to-the-network)

---

::: fastmob.network.haversine_m_batch
    options:
      show_source: false

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

---

::: fastmob.network.match_trajectory
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Match a GPS trace to roads](../learn/road-network-analysis.md#match-a-gps-trace-to-roads)

