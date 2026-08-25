"""Trajectory map matching over a prepared :class:`RoadNetwork`.

The matcher intentionally uses the graph's directed endpoint segments.  This
keeps it compatible with graphs built today; Overture polyline geometry can be
added later without changing the public result contract.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import cos, exp, inf, pi, sqrt

import narwhals as nw

from fastmob.core.base import unwrap_native
import pyarrow as pa

from fastmob.utils._common import _detect_trajectory_columns, _pick_existing_column, UID_CANDIDATES


def _seconds(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    try:
        return float(value) / (1000.0 if abs(float(value)) > 10_000_000_000 else 1.0)
    except (TypeError, ValueError):
        return None


def _project(lat: float, lng: float, edge: tuple[int, int, float, float, float, float, float]):
    """Project a coordinate on one local, endpoint-defined road segment."""
    start, end, a_lat, a_lng, b_lat, b_lng, length_m = edge
    scale = cos((a_lat + b_lat) * pi / 360.0)
    x, y = (lng - a_lng) * 111_320.0 * scale, (lat - a_lat) * 110_540.0
    dx, dy = (b_lng - a_lng) * 111_320.0 * scale, (b_lat - a_lat) * 110_540.0
    denom = dx * dx + dy * dy
    fraction = 0.0 if denom == 0 else min(1.0, max(0.0, (x * dx + y * dy) / denom))
    p_lat, p_lng = a_lat + fraction * (b_lat - a_lat), a_lng + fraction * (b_lng - a_lng)
    distance_m = sqrt((x - fraction * dx) ** 2 + (y - fraction * dy) ** 2)
    return start, end, fraction, p_lat, p_lng, distance_m, length_m


def _candidates(lat, lng, edges, radius_m, max_candidates, sigma_m):
    if lat is None or lng is None:
        return []
    result = [_project(float(lat), float(lng), edge) for edge in edges]
    result = [candidate for candidate in result if candidate[5] <= radius_m]
    result.sort(key=lambda candidate: (candidate[5], candidate[0], candidate[1]))
    return [(candidate, -(candidate[5] / sigma_m) ** 2 / 2.0) for candidate in result[:max_candidates]]


def _match_user(rows, edges, network, radius_m, max_candidates, sigma_m, gap_seconds, max_speed_mps):
    output = {row_index: (None, None, None, None, None, None, "unmatched", None) for row_index, *_ in rows}
    layers, scores, backpointers, row_indices = [], [], [], []
    previous_time = None

    def flush():
        if not layers:
            return
        state = max(range(len(scores[-1])), key=lambda index: scores[-1][index])
        chosen = [state]
        for pointers in reversed(backpointers[1:]):
            state = pointers[state]
            chosen.append(state)
        chosen.reverse()
        for layer_index, candidate_index in enumerate(chosen):
            candidate = layers[layer_index][candidate_index][0]
            ranked = sorted(scores[layer_index], reverse=True)
            margin = ranked[0] - ranked[1] if len(ranked) > 1 else 20.0
            output[row_indices[layer_index]] = (
                *candidate[:6],
                "gap_start" if layer_index == 0 else "matched",
                1.0 / (1.0 + exp(-margin)),
            )

    for row_index, lat, lng, timestamp in rows:
        current = _candidates(lat, lng, edges, radius_m, max_candidates, sigma_m)
        time = _seconds(timestamp)
        split = not layers or time is None or previous_time is None or time <= previous_time or time - previous_time > gap_seconds
        if not current:
            flush()
            layers, scores, backpointers, row_indices = [], [], [], []
            previous_time = time
            continue
        if split:
            flush()
            layers, scores, backpointers, row_indices = [current], [[emission for _candidate, emission in current]], [[-1] * len(current)], [row_index]
        else:
            # The routing kernel receives the whole candidate cross-product in
            # one Arrow batch.  It releases the GIL and parallelises queries.
            previous = layers[-1]
            from_nodes = [left[0][1] for left in previous for _right in current]
            to_nodes = [right[0][0] for _left in previous for right in current]
            distances, connected = network.batch_distances(from_nodes, to_nodes)
            distances, connected = distances.to_pylist(), connected.to_pylist()
            elapsed = time - previous_time
            layer_scores, pointers = [], []
            for right_idx, (right, emission) in enumerate(current):
                best_score, best_left = -inf, -1
                for left_idx, (_left, _emission) in enumerate(previous):
                    route_idx = left_idx * len(current) + right_idx
                    if not connected[route_idx]:
                        continue
                    route_m = distances[route_idx] + (1.0 - _left[2]) * _left[6] + right[2] * right[6]
                    if route_m / max(elapsed, 1.0) > max_speed_mps:
                        continue
                    # Prefer the shortest feasible routed transition.  The
                    # speed cap above rules out teleporting while this term
                    # resolves equal-distance GPS emissions at junctions.
                    score = scores[-1][left_idx] + emission - route_m / 1000.0
                    if score > best_score:
                        best_score, best_left = score, left_idx
                layer_scores.append(best_score)
                pointers.append(best_left)
            if max(layer_scores) == -inf:
                flush()
                layers, scores, backpointers, row_indices = [current], [[emission for _candidate, emission in current]], [[-1] * len(current)], [row_index]
            else:
                layers.append(current)
                scores.append(layer_scores)
                backpointers.append(pointers)
                row_indices.append(row_index)
        previous_time = time
    flush()
    return output


def match_trajectory(
    traj,
    *,
    network,
    uid_col: str | None = None,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    candidate_radius_m: float = 75.0,
    max_candidates: int = 8,
    gps_sigma_m: float = 20.0,
    gap_seconds: float = 1800.0,
    max_transition_speed_kmh: float = 180.0,
) -> pa.Table:
    """Map-match a trajectory with a directed-edge HMM/Viterbi model.

    Results are Arrow-native and preserve input row order. Independent user
    trajectories are evaluated concurrently; a user sequence is sequential.
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    datetime_col, lat_col, lng_col, detected_uid = _detect_trajectory_columns(
        df, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col, uid_col=uid_col
    )
    uid_col = uid_col or detected_uid or _pick_existing_column(df.columns, UID_CANDIDATES)
    table = df.to_arrow()
    node_coords = {row["node_idx"]: (row["lat"], row["lng"]) for row in network.nodes_df.to_pylist()}
    edges = [
        (row["from_node"], row["to_node"], *node_coords[row["from_node"]], *node_coords[row["to_node"]], row["length_m"])
        for row in network.edges_df.to_pylist() if row["from_node"] in node_coords and row["to_node"] in node_coords
    ]
    groups: dict[object, list[tuple[int, object, object, object]]] = {}
    for i, row in enumerate(table.select([lat_col, lng_col, datetime_col]).to_pylist()):
        key = table.column(uid_col)[i].as_py() if uid_col else 0
        groups.setdefault(key, []).append((i, row[lat_col], row[lng_col], row[datetime_col]))
    workers = min(32, max(1, len(groups)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = executor.map(
            lambda group: _match_user(group, edges, network, candidate_radius_m, max_candidates, gps_sigma_m, gap_seconds, max_transition_speed_kmh / 3.6),
            groups.values(),
        )
    result = [(None, None, None, None, None, None, "unmatched", None) for _ in range(table.num_rows)]
    for part in parts:
        for index, value in part.items():
            result[index] = value
    columns = {name: table.column(name) for name in table.column_names}
    names = ["matched_edge_from", "matched_edge_to", "matched_fraction", "matched_lat", "matched_lng", "distance_to_road_m", "match_status", "match_score"]
    for column, name in enumerate(names):
        columns[name] = pa.array([value[column] for value in result])
    return pa.table(columns)
