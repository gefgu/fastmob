# Route and map-match trajectories

Use the network module when a straight-line distance is not a useful proxy for
travel. Build the graph once, then reuse it for snapping, route queries, and
network-aware measures.

```python
from fastmob.network import RoadNetwork, fetch_road_network, match_trajectory
from fastmob.measures.individual import jump_lengths_road

nodes, edges = fetch_road_network(
    min_lon=2.34, min_lat=48.85, max_lon=2.36, max_lat=48.86,
    overture_release="2026-05-20.0",
)
network = RoadNetwork.build(edges, nodes)

road_jumps = jump_lengths_road(traj, network=network)
matched = match_trajectory(traj, network=network)
```

The graph build is reused; do not fetch or prepare it once per trajectory.
Snapping returns `-1` for a point outside the chosen threshold, and
network-aware distance measures fall back to Haversine when an endpoint is
unsnapped or disconnected.

`match_trajectory` returns one result row per input observation with matched
edge information, projected coordinates, status, and a score. It processes
users independently and splits tracks at invalid observations or large time
gaps. Its current edge projection uses connector-to-connector segments, so
inspect results carefully on curved roads.

!!! warning "Optional dependencies and data access"

    Fetching Overture networks requires the `network` extra and access to the
    requested Overture release. Save the node/edge tables with
    `build_road_graph` when repeatedly analysing the same area.

See the [network reference](../reference/network.md) for Arrow return types,
route geometry, OD desire lines, and rail support.

<!-- Visual placeholder: GPS points → candidate road edges → matched route map. -->
