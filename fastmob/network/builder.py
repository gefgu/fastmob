"""Build a routable road or rail graph from Overture Maps transportation data.

Fetches ``theme=transportation/type=segment`` for a bbox, splits each
segment's geometry into pieces between consecutive connectors (the routing
decision points along the segment), and turns those pieces into directed
graph edges weighted by travel time (roads: 80% of the speed limit, falling
back to a class-based default; rail: a fixed class-based speed).

Connector coordinates are derived by linearly interpolating each segment's
own geometry at the connector's ``at`` fraction, so a separate
``type=connector`` fetch isn't needed just to place nodes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .speeds import (
    CAR_SPEED_FACTOR,
    DEFAULT_RAIL_CLASSES,
    DEFAULT_RAIL_SPEED_KMH_BY_CLASS,
    DEFAULT_SPEED_KMH_BY_CLASS,
    DRIVABLE_CLASSES,
)

logger = logging.getLogger(__name__)

_MPH_TO_KMH = 1.609344


def _is_missing_or_empty(value) -> bool:
    """True for None/NaN/pd.NA and for empty lists/arrays.

    DuckDB NULL LIST columns can surface as ``None``, ``float('nan')``, or
    ``pd.NA`` depending on the pandas conversion path, none of which are
    safely truthy-checkable directly (``bool(pd.NA)`` raises).
    """
    if value is None:
        return True
    if not isinstance(value, (list, tuple)):
        return True  # a scalar missing-value sentinel (NaN/pd.NA), not a real list
    return len(value) == 0


def _speed_kmh_for_pair(speed_limits, road_class: str, from_at: float, to_at: float) -> float:
    default = DEFAULT_SPEED_KMH_BY_CLASS.get(road_class, DEFAULT_SPEED_KMH_BY_CLASS["residential"])
    if _is_missing_or_empty(speed_limits):
        return default

    mid = (from_at + to_at) / 2.0
    whole_segment_kmh: float | None = None
    for entry in speed_limits:
        max_speed = entry.get("max_speed") if entry else None
        if not max_speed or max_speed.get("value") is None:
            continue
        value = float(max_speed["value"])
        unit = (max_speed.get("unit") or "km/h").lower()
        kmh = value * _MPH_TO_KMH if unit == "mph" else value
        between = entry.get("between")
        if between and len(between) == 2:
            lo, hi = float(between[0]), float(between[1])
            if lo <= mid <= hi:
                return kmh
        elif whole_segment_kmh is None:
            whole_segment_kmh = kmh
    return whole_segment_kmh if whole_segment_kmh is not None else default


_NON_CAR_MODES = {"foot", "bicycle", "pedestrian"}


def _direction_for_pair(access_restrictions) -> str | None:
    """Return 'both', 'forward', 'backward', or None (not drivable)."""
    denied_forward = False
    denied_backward = False
    if _is_missing_or_empty(access_restrictions):
        access_restrictions = []
    for entry in access_restrictions:
        if not entry or entry.get("access_type") != "denied":
            continue
        when = entry.get("when") or {}
        mode = when.get("mode")
        if mode and all(str(m).lower() in _NON_CAR_MODES for m in mode):
            continue  # foot/bicycle-only restriction, irrelevant to car routing
        heading = when.get("heading")
        if heading == "forward":
            denied_forward = True
        elif heading == "backward":
            denied_backward = True
        elif heading is None:
            denied_forward = True
            denied_backward = True
    if denied_forward and denied_backward:
        return None
    if denied_forward:
        return "backward"
    if denied_backward:
        return "forward"
    return "both"


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without duckdb installed
        raise ImportError(
            "fetching a road/rail network requires duckdb. Install it with `pip install fastmob[network]`."
        ) from exc
    return duckdb


def fetch_road_network(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and build a car-routable graph from Overture road segments.

    Returns
    -------
    nodes_df, edges_df:
        ``nodes_df``: ``node_idx`` (dense, 0-based), ``connector_id``,
        ``lat``, ``lng``. ``edges_df``: ``from_node``, ``to_node``,
        ``length_m``, ``speed_kmh``, ``weight_ds``, ``class``.
    """
    duckdb = _require_duckdb()

    class_list = ", ".join(f"'{c}'" for c in DRIVABLE_CLASSES)
    logger.info("Fetching Overture Maps %s road segments ...", overture_release)
    df = duckdb.sql(f"""
        INSTALL spatial;  LOAD spatial;
        SET s3_region = 'us-west-2';

        WITH segs AS (
            SELECT
                id AS segment_id,
                class,
                speed_limits,
                access_restrictions,
                geometry,
                UNNEST(connectors) AS conn
            FROM read_parquet(
                's3://overturemaps-us-west-2/release/{overture_release}/theme=transportation/type=segment/*',
                filename=true, hive_partitioning=1
            )
            WHERE subtype = 'road'
              AND class IN ({class_list})
              AND bbox.xmin BETWEEN {min_lon} AND {max_lon}
              AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
        ),
        ordered AS (
            SELECT
                segment_id, class, speed_limits, access_restrictions, geometry,
                conn.connector_id AS connector_id,
                conn.at AS at,
                ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY conn.at) AS rn
            FROM segs
        ),
        pairs AS (
            SELECT
                o1.segment_id,
                o1.class AS road_class,
                o1.speed_limits,
                o1.access_restrictions,
                o1.connector_id AS from_connector,
                o1.at AS from_at,
                o2.connector_id AS to_connector,
                o2.at AS to_at,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lng,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lng
            FROM ordered o1
            JOIN ordered o2 ON o1.segment_id = o2.segment_id AND o2.rn = o1.rn + 1
        )
        SELECT
            road_class, speed_limits, access_restrictions,
            from_connector, from_at, to_connector, to_at,
            from_lat, from_lng, to_lat, to_lng,
            2 * 6371000 * ASIN(SQRT(
                POWER(SIN(RADIANS((to_lat - from_lat) / 2)), 2) +
                COS(RADIANS(from_lat)) * COS(RADIANS(to_lat)) *
                POWER(SIN(RADIANS((to_lng - from_lng) / 2)), 2)
            )) AS length_m
        FROM pairs
    """).df()

    logger.info("Fetched %d road segment pieces; deriving speeds/direction ...", len(df))

    node_coords: dict[str, tuple[float, float]] = {}
    records: list[tuple[str, str, float, float, int, str]] = []

    for row in df.itertuples(index=False):
        speed_kmh = _speed_kmh_for_pair(row.speed_limits, row.road_class, row.from_at, row.to_at)
        direction = _direction_for_pair(row.access_restrictions)
        if direction is None or speed_kmh <= 0:
            continue
        length_m = float(row.length_m) if row.length_m and row.length_m > 0 else 0.1
        speed_mps = (speed_kmh * CAR_SPEED_FACTOR) / 3.6
        weight_ds = max(1, round(length_m / speed_mps * 10))
        node_coords[row.from_connector] = (row.from_lat, row.from_lng)
        node_coords[row.to_connector] = (row.to_lat, row.to_lng)
        if direction in ("both", "forward"):
            records.append((row.from_connector, row.to_connector, length_m, speed_kmh, weight_ds, row.road_class))
        if direction in ("both", "backward"):
            records.append((row.to_connector, row.from_connector, length_m, speed_kmh, weight_ds, row.road_class))

    connector_ids = sorted(node_coords)
    connector_to_idx = {cid: i for i, cid in enumerate(connector_ids)}
    nodes_df = pd.DataFrame(
        {
            "node_idx": np.arange(len(connector_ids), dtype=np.int64),
            "connector_id": connector_ids,
            "lat": [node_coords[c][0] for c in connector_ids],
            "lng": [node_coords[c][1] for c in connector_ids],
        }
    )

    edges_df = pd.DataFrame(
        records,
        columns=["from_connector", "to_connector", "length_m", "speed_kmh", "weight_ds", "class"],
    )
    edges_df["from_node"] = edges_df["from_connector"].map(connector_to_idx).astype(np.int64)
    edges_df["to_node"] = edges_df["to_connector"].map(connector_to_idx).astype(np.int64)
    edges_df = edges_df[["from_node", "to_node", "length_m", "speed_kmh", "weight_ds", "class"]]

    return nodes_df, edges_df


def build_road_graph(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    nodes_output: str,
    edges_output: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a cached road graph from disk, or fetch and cache it."""
    if Path(nodes_output).exists() and Path(edges_output).exists():
        logger.info("Loading cached road graph from %s / %s ...", nodes_output, edges_output)
        return pd.read_parquet(nodes_output), pd.read_parquet(edges_output)

    nodes_df, edges_df = fetch_road_network(min_lon, min_lat, max_lon, max_lat, overture_release)
    Path(nodes_output).parent.mkdir(parents=True, exist_ok=True)
    Path(edges_output).parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(nodes_output, index=False)
    edges_df.to_parquet(edges_output, index=False)
    logger.info(
        "Saved road graph: %d nodes, %d directed edges -> %s, %s",
        len(nodes_df),
        len(edges_df),
        nodes_output,
        edges_output,
    )
    return nodes_df, edges_df


def fetch_rail_network(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    classes: list[str] | None = None,
    speed_kmh_by_class: dict[str, float] | None = None,
    default_speed_kmh: float = 35.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and build a simple bidirectional rail graph from Overture segments."""
    duckdb = _require_duckdb()

    rail_classes = classes or DEFAULT_RAIL_CLASSES
    speed_by_class = speed_kmh_by_class or DEFAULT_RAIL_SPEED_KMH_BY_CLASS
    class_list = ", ".join(f"'{c}'" for c in rail_classes)
    logger.info("Fetching Overture Maps %s rail segments ...", overture_release)
    df = duckdb.sql(f"""
        INSTALL spatial;  LOAD spatial;
        SET s3_region = 'us-west-2';

        WITH segs AS (
            SELECT
                id AS segment_id,
                class,
                geometry,
                UNNEST(connectors) AS conn
            FROM read_parquet(
                's3://overturemaps-us-west-2/release/{overture_release}/theme=transportation/type=segment/*',
                filename=true, hive_partitioning=1
            )
            WHERE subtype = 'rail'
              AND class IN ({class_list})
              AND bbox.xmin BETWEEN {min_lon} AND {max_lon}
              AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
        ),
        ordered AS (
            SELECT
                segment_id, class, geometry,
                conn.connector_id AS connector_id,
                conn.at AS at,
                ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY conn.at) AS rn
            FROM segs
        ),
        pairs AS (
            SELECT
                o1.segment_id,
                o1.class AS rail_class,
                o1.connector_id AS from_connector,
                o1.at AS from_at,
                o2.connector_id AS to_connector,
                o2.at AS to_at,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lng,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lng
            FROM ordered o1
            JOIN ordered o2 ON o1.segment_id = o2.segment_id AND o2.rn = o1.rn + 1
        )
        SELECT
            rail_class, from_connector, from_at, to_connector, to_at,
            from_lat, from_lng, to_lat, to_lng,
            2 * 6371000 * ASIN(SQRT(
                POWER(SIN(RADIANS((to_lat - from_lat) / 2)), 2) +
                COS(RADIANS(from_lat)) * COS(RADIANS(to_lat)) *
                POWER(SIN(RADIANS((to_lng - from_lng) / 2)), 2)
            )) AS length_m
        FROM pairs
    """).df()

    logger.info("Fetched %d rail segment pieces; deriving weights ...", len(df))
    node_coords: dict[str, tuple[float, float]] = {}
    records: list[tuple[str, str, float, float, int, str]] = []
    for row in df.itertuples(index=False):
        speed_kmh = float(speed_by_class.get(row.rail_class, default_speed_kmh))
        if speed_kmh <= 0:
            continue
        length_m = float(row.length_m) if row.length_m and row.length_m > 0 else 0.1
        speed_mps = speed_kmh / 3.6
        weight_ds = max(1, round(length_m / speed_mps * 10))
        node_coords[row.from_connector] = (row.from_lat, row.from_lng)
        node_coords[row.to_connector] = (row.to_lat, row.to_lng)
        records.append((row.from_connector, row.to_connector, length_m, speed_kmh, weight_ds, row.rail_class))
        records.append((row.to_connector, row.from_connector, length_m, speed_kmh, weight_ds, row.rail_class))

    connector_ids = sorted(node_coords)
    connector_to_idx = {cid: i for i, cid in enumerate(connector_ids)}
    nodes_df = pd.DataFrame(
        {
            "node_idx": np.arange(len(connector_ids), dtype=np.int64),
            "connector_id": connector_ids,
            "lat": [node_coords[c][0] for c in connector_ids],
            "lng": [node_coords[c][1] for c in connector_ids],
        }
    )
    edges_df = pd.DataFrame(
        records,
        columns=["from_connector", "to_connector", "length_m", "speed_kmh", "weight_ds", "class"],
    )
    if len(edges_df) == 0:
        edges_df = pd.DataFrame(columns=["from_node", "to_node", "length_m", "speed_kmh", "weight_ds", "class"])
    else:
        edges_df["from_node"] = edges_df["from_connector"].map(connector_to_idx).astype(np.int64)
        edges_df["to_node"] = edges_df["to_connector"].map(connector_to_idx).astype(np.int64)
        edges_df = edges_df[["from_node", "to_node", "length_m", "speed_kmh", "weight_ds", "class"]]
    return nodes_df, edges_df


def build_rail_graph(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    nodes_output: str,
    edges_output: str,
    classes: list[str] | None = None,
    speed_kmh_by_class: dict[str, float] | None = None,
    default_speed_kmh: float = 35.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a cached rail graph from disk, or fetch and cache it."""
    if Path(nodes_output).exists() and Path(edges_output).exists():
        logger.info("Loading cached rail graph from %s / %s ...", nodes_output, edges_output)
        return pd.read_parquet(nodes_output), pd.read_parquet(edges_output)

    nodes_df, edges_df = fetch_rail_network(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        overture_release,
        classes,
        speed_kmh_by_class,
        default_speed_kmh,
    )
    Path(nodes_output).parent.mkdir(parents=True, exist_ok=True)
    Path(edges_output).parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(nodes_output, index=False)
    edges_df.to_parquet(edges_output, index=False)
    logger.info(
        "Saved rail graph: %d nodes, %d directed edges -> %s, %s",
        len(nodes_df),
        len(edges_df),
        nodes_output,
        edges_output,
    )
    return nodes_df, edges_df
