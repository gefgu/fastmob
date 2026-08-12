# Network

Road/rail-network-constrained distance: build a routable graph from
Overture Maps transportation data, snap trajectory points to it, and query
network (not straight-line) distance via a Rust contraction-hierarchy router.

| API | Description |
| --- | --- |
| [`RoadNetwork`](#fastmob.network.RoadNetwork) | A road/rail network prepared once (contraction hierarchy), reused for many distance queries. |
| [`fetch_road_network`](#fastmob.network.fetch_road_network) | Fetch and build a car-routable graph from Overture road segments. |
| [`build_road_graph`](#fastmob.network.build_road_graph) | Load a cached road graph from disk, or fetch and cache it. |
| [`fetch_rail_network`](#fastmob.network.fetch_rail_network) | Fetch and build a bidirectional rail graph from Overture segments. |
| [`build_rail_graph`](#fastmob.network.build_rail_graph) | Load a cached rail graph from disk, or fetch and cache it. |
| [`snap_locations_to_graph`](#fastmob.network.snap_locations_to_graph) | Snap each row to its nearest road/rail graph node. |
| [`RoadNetwork.batch_routes`](#fastmob.network.RoadNetwork.batch_routes) | Route geometry (waypoint coordinates) for `(from_node, to_node)` queries. |
| [`od_desire_lines`](#fastmob.network.od_desire_lines) | Aggregate OD-pair flows onto graph edges (desire lines). |
| [`haversine_m_batch`](#fastmob.network.haversine_m_batch) | Vectorized Haversine distance (metres) between two arrays of points. |

`fetch_road_network` and `fetch_rail_network` require DuckDB: install it with
`pip install duckdb`. Snapping uses Fastmob's native Rust spatial index and
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

---

::: fastmob.network.fetch_road_network
    options:
      show_source: false

---

::: fastmob.network.build_road_graph
    options:
      show_source: false

---

::: fastmob.network.fetch_rail_network
    options:
      show_source: false

---

::: fastmob.network.build_rail_graph
    options:
      show_source: false

---

::: fastmob.network.snap_locations_to_graph
    options:
      show_source: false

---

::: fastmob.network.haversine_m_batch
    options:
      show_source: false

---

::: fastmob.network.od_desire_lines
    options:
      show_source: false

---

::: fastmob.measures.individual.network_distance.jump_lengths_road
    options:
      show_source: false

---

::: fastmob.measures.individual.network_distance.radius_of_gyration_road
    options:
      show_source: false
