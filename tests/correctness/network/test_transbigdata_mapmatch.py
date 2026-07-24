"""Comparison test: fastmob's nearest-node snap vs TransBigData's traj_mapmatch.

fastmob's map-matching capability (:func:`fastmob.network.snap_locations_to_graph`)
snaps each point to its nearest graph *node*. TransBigData's
``traj_mapmatch`` snaps each point to the nearest position along the
nearest graph *edge* (interpolating along the edge's LineString) -- a
strictly-at-least-as-good projection, since a node is itself always one
point along its incident edges. This test checks the two approaches agree
in identifying the correct local edge/node, and that fastmob's node-snap
distance is never smaller than TransBigData's edge-interpolated distance
for the same point (the edge projection can only be closer or equal, never
farther).

See ``tests/populate_transbigdata_cache.py`` for how the cached synthetic
network + reference match was generated.
"""

from __future__ import annotations

import pytest
from fastmob.network.snap import snap_locations_to_graph

pytestmark = pytest.mark.transbigdata


def test_fastmob_node_snap_distance_is_at_least_transbigdata_edge_snap(transbigdata_reference):
    nodes_df = transbigdata_reference.nodes()
    input_df = transbigdata_reference.input_traj()
    matched_df = transbigdata_reference.matched()

    node_idx = snap_locations_to_graph(input_df, nodes_df, max_distance_m=1_000_000.0)
    assert (node_idx >= 0).all()

    from fastmob.network._util import haversine_m_batch

    snapped_lat = nodes_df.set_index("node_idx").loc[node_idx, "lat"].to_numpy()
    snapped_lng = nodes_df.set_index("node_idx").loc[node_idx, "lng"].to_numpy()
    fastmob_dist_m = haversine_m_batch(input_df["lat"].to_numpy(), input_df["lng"].to_numpy(), snapped_lat, snapped_lng)

    # A small epsilon absorbs the same haversine-kernel-precision slack
    # CLAUDE.md documents for skmob distance comparisons (different
    # haversine implementations, not a correctness gap).
    assert (fastmob_dist_m >= matched_df["dist_m"].to_numpy() - 1.0).all()


def test_fastmob_snaps_to_an_endpoint_of_transbigdatas_matched_edge(transbigdata_reference):
    # Toy grid: matched point 0 lands on edge 0->1 near node 0; matched
    # point 1 lands on edge 2->3 near node 3 (see populate script). fastmob's
    # nearest-node snap should pick one of that same edge's two endpoints.
    nodes_df = transbigdata_reference.nodes()
    input_df = transbigdata_reference.input_traj()

    node_idx = snap_locations_to_graph(input_df, nodes_df, max_distance_m=1_000_000.0)
    assert node_idx[0] in (0, 1)
    assert node_idx[1] in (2, 3)
