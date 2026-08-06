from __future__ import annotations

import narwhals as nw
import pyarrow as pa

def snap_locations_to_graph(
    tessellation_df,
    nodes_df,
    max_distance_m: float,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> pa.Array:
    """Snap each tessellation row to its nearest road/rail graph node.

    Parameters
    ----------
    tessellation_df:
        Rows with lat/lng columns to snap.
    nodes_df:
        Graph nodes with columns ``node_idx``, ``lat``, ``lng`` (as returned
        by :func:`fastmob.network.builder.fetch_road_network` / `fetch_rail_network`).
    max_distance_m:
        Maximum snap distance; farther rows are reported unsnapped.
    lat_col, lng_col:
        Column names on ``tessellation_df``.

    Returns
    -------
    pyarrow.Int64Array
        int64 array aligned 1:1 with ``tessellation_df`` rows; ``-1`` when
        the nearest node is farther than ``max_distance_m`` (unsnapped).
    """
    nodes = nw.from_native(nodes_df, eager_only=True)
    tess = nw.from_native(tessellation_df, eager_only=True)
    if len(tess) == 0:
        return pa.array([], type=pa.int64())
    if len(nodes) == 0:
        return pa.array([-1] * len(tess), type=pa.int64())

    from fastmob._core import nearest_nodes_arrow

    nearest_row, _distance_m = nearest_nodes_arrow(
        tess.get_column(lat_col).to_arrow(),
        tess.get_column(lng_col).to_arrow(),
        nodes.get_column("lat").to_arrow(),
        nodes.get_column("lng").to_arrow(),
        max_distance_m,
    )
    node_ids = nodes.get_column("node_idx").to_arrow()
    # Rust returns positions into `nodes`; node_idx need not be row order.
    return pa.array(
        [node_ids[row].as_py() if row >= 0 else -1 for row in nearest_row.to_pylist()],
        type=pa.int64(),
    )
