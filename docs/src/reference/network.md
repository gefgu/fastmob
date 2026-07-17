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
| [`haversine_m_batch`](#fastmob.network.haversine_m_batch) | Vectorized Haversine distance (metres) between two arrays of points. |

Requires the `network` extra (`pip install fastmob[network]`) for
`fetch_road_network`/`fetch_rail_network` (needs `duckdb`) and the `ai`
extra for `snap_locations_to_graph` (needs `scikit-learn`).

## Network-aware distance measures

`fastmob.measures.individual.jump_lengths_km`/`radius_of_gyration_km` mirror
`jump_lengths`/`radius_of_gyration` but measure distance along a prepared
`RoadNetwork` instead of straight-line, falling back to Haversine per-pair
wherever a point is unsnapped or the graph is disconnected between the two
points:

```python
import pandas as pd
from fastmob.network import RoadNetwork, fetch_road_network
from fastmob.measures.individual import jump_lengths_km, radius_of_gyration_km

nodes_df, edges_df = fetch_road_network(2.34, 48.85, 2.36, 48.86, "2026-05-20.0")
network = RoadNetwork.build(edges_df, nodes_df)

traj = pd.DataFrame(...)  # uid, datetime, lat, lng columns
jumps_km = jump_lengths_km(traj, network=network)
rg_km = radius_of_gyration_km(traj, network=network)
```

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

::: fastmob.measures.individual.network_distance.jump_lengths_km
    options:
      show_source: false

---

::: fastmob.measures.individual.network_distance.radius_of_gyration_km
    options:
      show_source: false
