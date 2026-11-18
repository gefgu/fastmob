"""Build a routable road or rail graph from Overture Maps transportation data.

Fetches ``theme=transportation/type=segment`` for a bbox, splits each
segment's geometry into pieces between consecutive connectors (the routing
decision points along the segment), and turns those pieces into directed
graph edges weighted by travel time (car roads: 80% of the speed limit, falling
back to a class-based default; walking: a fixed pace; rail: a fixed class-based speed).

Connector coordinates are derived by linearly interpolating each segment's
own geometry at the connector's ``at`` fraction, so a separate
``type=connector`` fetch isn't needed just to place nodes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .speeds import (
    CAR_SPEED_FACTOR,
    DEFAULT_RAIL_CLASSES,
    DEFAULT_RAIL_SPEED_KMH_BY_CLASS,
    DEFAULT_SPEED_KMH_BY_CLASS,
    DEFAULT_WALK_SPEED_KMH,
    DRIVABLE_CLASSES,
    WALKABLE_CLASSES,
)

logger = logging.getLogger(__name__)

_MPH_TO_KMH = 1.609344


def _is_missing_or_empty(value) -> bool:
    """True for None/NaN and for empty lists/arrays.

    DuckDB NULL LIST columns surface as plain Python ``None`` when read via
    ``.arrow()`` + ``to_pylist()``. The ``float('nan')`` branch is kept
    defensively for any other scalar missing-value sentinel a caller might
    pass in directly (``bool(float('nan'))`` is truthy, so it can't be
    truthy-checked without this).
    """
    if value is None:
        return True
    if not isinstance(value, (list, tuple)):
        return True  # a scalar missing-value sentinel (e.g. NaN), not a real list
    return len(value) == 0


def _speed_kmh_for_pair(
    speed_limits,
    road_class: str,
    from_at: float,
    to_at: float,
    default_by_class: dict[str, float] | None = None,
) -> float:
    table = default_by_class or DEFAULT_SPEED_KMH_BY_CLASS
    default = table.get(road_class, table.get("residential", DEFAULT_SPEED_KMH_BY_CLASS["residential"]))
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


_FOOT_MODES = ("foot", "pedestrian")


def _walk_direction(access_restrictions) -> str | None:
    """Return 'both', 'forward', 'backward', or None (not walkable) from a piece's access restrictions.

    Only unconditional restrictions count (those with ``during``, ``using``, ``vehicle`` or ``recognized``
    conditions apply only sometimes, so they do not close the piece). A restriction applies to pedestrians when it
    names no mode or names foot. An explicit ``allowed`` or ``designated`` entry for foot overrides denials
    (a pedestrian-only street is ``denied`` for all modes and ``allowed`` for foot).
    """
    if _is_missing_or_empty(access_restrictions):
        return "both"
    denied_forward = denied_backward = False
    for entry in access_restrictions:
        if not entry:
            continue
        when = entry.get("when") or {}
        if any(when.get(key) for key in ("during", "using", "vehicle", "recognized")):
            continue
        modes = [str(m).lower() for m in (when.get("mode") or [])]
        names_foot = any(m in _FOOT_MODES for m in modes)
        if entry.get("access_type") in ("allowed", "designated") and names_foot:
            return "both"
        if entry.get("access_type") != "denied" or (modes and not names_foot):
            continue
        heading = when.get("heading")
        denied_forward = denied_forward or heading in (None, "forward")
        denied_backward = denied_backward or heading in (None, "backward")
    if denied_forward and denied_backward:
        return None
    if denied_forward:
        return "backward"
    if denied_backward:
        return "forward"
    return "both"


def _segment_source(overture_release: str) -> str:
    """The Parquet glob of the Overture transportation segments of a release (replaced by a local file in tests)."""
    return f"s3://overturemaps-us-west-2/release/{overture_release}/theme=transportation/type=segment/*"


def _fetch_pieces(duckdb, overture_release, subtype, classes, bbox, road_fields):
    """Pieces of the Overture segments between consecutive connectors, one row per piece.

    ``length_m`` is the length along the segment between the two connectors (the geometry cut at their ``at``
    fractions, measured on the spheroid), so a curved piece is not shortened to the straight line.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if not classes:
        raise ValueError("classes must name at least one road class")
    class_list = ", ".join("'" + str(c).replace("'", "''") + "'" for c in classes)
    extra = "speed_limits, access_restrictions," if road_fields else ""
    carried = "o1.speed_limits, o1.access_restrictions," if road_fields else ""
    return duckdb.sql(f"""
        INSTALL spatial;  LOAD spatial;
        SET s3_region = 'us-west-2';

        WITH segs AS (
            SELECT
                id AS segment_id,
                names.primary AS name,
                class,
                {extra}
                geometry,
                UNNEST(connectors) AS conn
            FROM read_parquet(
                '{_segment_source(overture_release)}',
                filename=true, hive_partitioning=1
            )
            WHERE subtype = '{subtype}'
              AND class IN ({class_list})
              AND bbox.xmin BETWEEN {min_lon} AND {max_lon}
              AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
        ),
        ordered AS (
            SELECT
                segment_id, name, class, {extra} geometry,
                conn.connector_id AS connector_id,
                conn.at AS at,
                ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY conn.at) AS rn
            FROM segs
        ),
        pairs AS (
            SELECT
                o1.segment_id,
                o1.name,
                o1.class AS segment_class,
                {carried}
                o1.connector_id AS from_connector,
                o1.at AS from_at,
                o2.connector_id AS to_connector,
                o2.at AS to_at,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o1.at)) AS from_lng,
                ST_Y(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lat,
                ST_X(ST_LineInterpolatePoint(o1.geometry, o2.at)) AS to_lng,
                CASE WHEN o2.at > o1.at
                     THEN ST_Length_Spheroid(ST_FlipCoordinates(ST_LineSubstring(o1.geometry, o1.at, o2.at)))
                     ELSE 0 END AS length_m
            FROM ordered o1
            JOIN ordered o2 ON o1.segment_id = o2.segment_id AND o2.rn = o1.rn + 1
        )
        SELECT * FROM pairs
    """).to_arrow_table()


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without duckdb installed
        raise ImportError(
            "fetching a road/rail network requires duckdb. Install it with `pip install duckdb`."
        ) from exc
    return duckdb


def _nodes_table(node_coords: dict[str, tuple[float, float]]) -> tuple[pa.Table, dict[str, int]]:
    """Build the ``nodes_df`` table plus a connector-id -> dense-node-idx map."""
    connector_ids = sorted(node_coords)
    connector_to_idx = {cid: i for i, cid in enumerate(connector_ids)}
    nodes_df = pa.table(
        {
            "node_idx": pa.array(range(len(connector_ids)), type=pa.int64()),
            "connector_id": pa.array(connector_ids, type=pa.string()),
            "lat": pa.array([node_coords[c][0] for c in connector_ids], type=pa.float64()),
            "lng": pa.array([node_coords[c][1] for c in connector_ids], type=pa.float64()),
        }
    )
    return nodes_df, connector_to_idx


def _edges_table(
    records: list[tuple[str, str, float, float, int, str, str, str | None]],
    connector_to_idx: dict[str, int],
) -> pa.Table:
    """Build the ``edges_df`` table from ``(from_connector, to_connector, length_m,
    speed_kmh, weight_ds, class, segment_id, name)`` records, resolving connector ids to dense node ids.
    """
    return pa.table(
        {
            "from_node": pa.array([connector_to_idx[r[0]] for r in records], type=pa.int64()),
            "to_node": pa.array([connector_to_idx[r[1]] for r in records], type=pa.int64()),
            "length_m": pa.array([r[2] for r in records], type=pa.float64()),
            "speed_kmh": pa.array([r[3] for r in records], type=pa.float64()),
            "weight_ds": pa.array([r[4] for r in records], type=pa.int64()),
            "class": pa.array([r[5] for r in records], type=pa.string()),
            "segment_id": pa.array([r[6] for r in records], type=pa.string()),
            "name": pa.array([r[7] for r in records], type=pa.string()),
        }
    )


def fetch_road_network(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    mode: str = "drive",
    classes: list[str] | None = None,
    speed_kmh_by_class: dict[str, float] | None = None,
    directed: bool | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Fetch and build a routable graph from Overture road segments, for cars or for pedestrians.

    With ``mode="drive"`` (default) the graph is for cars: the drivable classes, travel time at 80% of the speed
    limit (or of the class speed when the segment has none) and one-way streets honoured. With ``mode="walk"``
    it is for pedestrians: footways, steps, paths and pedestrian streets plus the streets that can be walked
    (not motorways or trunk roads), edges usable both ways at a fixed walking speed of 5.04 km/h (1.4 m/s), leaving out
    pieces where access is denied to pedestrians (an explicit foot allowance, as on pedestrian-only streets, wins).

    Parameters
    ----------
    min_lon : float
        Western edge of the bounding box, in decimal degrees.
    min_lat : float
        Southern edge of the bounding box, in decimal degrees.
    max_lon : float
        Eastern edge of the bounding box, in decimal degrees.
    max_lat : float
        Northern edge of the bounding box, in decimal degrees.
    overture_release : str
        Overture Maps release to read, as listed in the public bucket, for
        example ``"2025-06-25.0"``. Overture removes old releases.
    mode : {"drive", "walk"}, optional
        The kind of network. Default ``"drive"``.
    classes : list of str, optional
        Overture road classes to keep. Default: the drivable classes for ``"drive"``, the walkable classes
        (``fastmob.network.speeds.WALKABLE_CLASSES``) for ``"walk"``.
    speed_kmh_by_class : dict of str to float, optional
        Speed in km/h for each class, which overrides the default. For ``"drive"`` it applies to segments that have
        no speed limit; for ``"walk"`` it replaces the walking speed of that class.
    directed : bool, optional
        Whether one-way (heading) restrictions are honoured: those that apply to cars for ``"drive"`` and those that
        apply to pedestrians for ``"walk"``. Default ``True`` for ``"drive"`` and ``False`` for ``"walk"``;
        ``False`` makes every edge usable in both directions.

    Returns
    -------
    nodes_df : pyarrow.Table
        ``node_idx`` (dense, 0-based), ``connector_id``, ``lat``, ``lng``.
    edges_df : pyarrow.Table
        ``from_node``, ``to_node``, ``length_m``, ``speed_kmh``, ``weight_ds``, ``class``, ``segment_id`` (the
        Overture segment the edge belongs to) and ``name`` (its primary street name, null when it has none). An
        edge is the piece of a segment between two connectors; ``length_m`` is measured along the street, so
        curved pieces are not shortened to the straight line. ``RoadNetwork.build`` ignores the extra columns.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"drive"`` or ``"walk"``.
    """
    if mode not in ("drive", "walk"):
        raise ValueError(f"mode must be 'drive' or 'walk', got {mode!r}")
    walk = mode == "walk"
    if directed is None:
        directed = not walk
    duckdb = _require_duckdb()

    road_classes = list(classes) if classes is not None else list(WALKABLE_CLASSES if walk else DRIVABLE_CLASSES)
    speeds = dict(speed_kmh_by_class or {})
    logger.info("Fetching Overture Maps %s road segments (%s) ...", overture_release, mode)
    df = _fetch_pieces(
        duckdb, overture_release, "road", road_classes, (min_lon, min_lat, max_lon, max_lat), road_fields=True
    )
    logger.info("Fetched %d road segment pieces; deriving speeds/direction ...", df.num_rows)

    node_coords: dict[str, tuple[float, float]] = {}
    records: list[tuple[str, str, float, float, int, str, str, str | None]] = []

    for row in df.to_pylist():
        road_class = row["segment_class"]
        if walk:
            direction = _walk_direction(row["access_restrictions"])
            if direction is not None and not directed:
                direction = "both"
            speed_kmh = float(speeds.get(road_class, DEFAULT_WALK_SPEED_KMH))
            speed_mps = speed_kmh / 3.6
        else:
            table = {**DEFAULT_SPEED_KMH_BY_CLASS, **speeds}
            speed_kmh = _speed_kmh_for_pair(
                row["speed_limits"], road_class, row["from_at"], row["to_at"], default_by_class=table
            )
            direction = _direction_for_pair(row["access_restrictions"])
            if direction is not None and not directed:
                direction = "both"
            speed_mps = (speed_kmh * CAR_SPEED_FACTOR) / 3.6
        if direction is None or speed_kmh <= 0:
            continue
        length_m = float(row["length_m"]) if row["length_m"] and row["length_m"] > 0 else 0.1
        weight_ds = max(1, round(length_m / speed_mps * 10))
        node_coords[row["from_connector"]] = (row["from_lat"], row["from_lng"])
        node_coords[row["to_connector"]] = (row["to_lat"], row["to_lng"])
        tail = (length_m, speed_kmh, weight_ds, road_class, row["segment_id"], row["name"])
        if direction in ("both", "forward"):
            records.append((row["from_connector"], row["to_connector"], *tail))
        if direction in ("both", "backward"):
            records.append((row["to_connector"], row["from_connector"], *tail))

    nodes_df, connector_to_idx = _nodes_table(node_coords)
    edges_df = _edges_table(records, connector_to_idx)

    return nodes_df, edges_df


def build_road_graph(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    overture_release: str,
    nodes_output: str,
    edges_output: str,
    mode: str = "drive",
    classes: list[str] | None = None,
    speed_kmh_by_class: dict[str, float] | None = None,
    directed: bool | None = None,
) -> tuple[pa.Table, pa.Table]:
    """Load a saved road graph from disk, or fetch it from Overture Maps and save it.

    If both ``nodes_output`` and ``edges_output`` exist they are read (no download); otherwise the network is
    fetched with ``fetch_road_network`` and written there as Parquet.

    Parameters
    ----------
    min_lon : float
        Bounding box in degrees (WGS 84).
    min_lat : float
        Bounding box in degrees (WGS 84).
    max_lon : float
        Bounding box in degrees (WGS 84).
    max_lat : float
        Bounding box in degrees (WGS 84).
    overture_release : str
        Overture Maps release id, for example ``"2026-10-21.0"``. Old releases are removed from the public bucket,
        so choose a current one.
    nodes_output : str
        Paths of the Parquet files for the graph nodes and edges.
    edges_output : str
        Paths of the Parquet files for the graph nodes and edges.
    mode : {"drive", "walk"}, optional
        As in ``fetch_road_network``. A saved graph is read back whatever ``mode`` is, so use different output
        paths for the drive and the walk network.
    classes : optional
        As in ``fetch_road_network``.
    speed_kmh_by_class : optional
        As in ``fetch_road_network``.
    directed : optional
        As in ``fetch_road_network``.

    Returns
    -------
    (nodes, edges) : tuple of pyarrow.Table
        As returned by ``fetch_road_network``: nodes with ``node_idx``, ``connector_id``, ``lat``, ``lng``; edges with
        ``from_node``, ``to_node``, ``length_m``, ``speed_kmh``, ``weight_ds`` (travel time in deciseconds), ``class``,
        ``segment_id`` and ``name``.
    """
    if Path(nodes_output).exists() and Path(edges_output).exists():
        logger.info("Loading cached road graph from %s / %s ...", nodes_output, edges_output)
        return pq.read_table(nodes_output), pq.read_table(edges_output)

    nodes_df, edges_df = fetch_road_network(
        min_lon, min_lat, max_lon, max_lat, overture_release, mode, classes, speed_kmh_by_class, directed
    )
    Path(nodes_output).parent.mkdir(parents=True, exist_ok=True)
    Path(edges_output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(nodes_df, nodes_output)
    pq.write_table(edges_df, edges_output)
    logger.info(
        "Saved road graph: %d nodes, %d directed edges -> %s, %s",
        nodes_df.num_rows,
        edges_df.num_rows,
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
) -> tuple[pa.Table, pa.Table]:
    """Fetch rail segments from Overture Maps and build a bidirectional rail graph.

    Parameters
    ----------
    min_lon : float
        Bounding box in degrees (WGS 84).
    min_lat : float
        Bounding box in degrees (WGS 84).
    max_lon : float
        Bounding box in degrees (WGS 84).
    max_lat : float
        Bounding box in degrees (WGS 84).
    overture_release : str
        Overture Maps release id, for example ``"2026-10-21.0"``. Old releases are removed from the public bucket,
        so choose a current one.
    classes : list of str, optional
        Overture rail classes to keep (default: light rail, standard gauge, subway and tram).
    speed_kmh_by_class : dict, optional
        Speed in km/h for each class, used to compute travel times.
    default_speed_kmh : float, optional
        Speed for a class not in ``speed_kmh_by_class``. Default 35.

    Returns
    -------
    (nodes, edges) : tuple of pyarrow.Table
        Same columns as ``fetch_road_network``; every segment appears in both directions. Rail and road graphs are
        separate and not connected.
    """
    duckdb = _require_duckdb()

    rail_classes = classes or DEFAULT_RAIL_CLASSES
    speed_by_class = speed_kmh_by_class or DEFAULT_RAIL_SPEED_KMH_BY_CLASS
    logger.info("Fetching Overture Maps %s rail segments ...", overture_release)
    df = _fetch_pieces(
        duckdb, overture_release, "rail", rail_classes, (min_lon, min_lat, max_lon, max_lat), road_fields=False
    )

    logger.info("Fetched %d rail segment pieces; deriving weights ...", df.num_rows)
    node_coords: dict[str, tuple[float, float]] = {}
    records: list[tuple[str, str, float, float, int, str, str, str | None]] = []
    for row in df.to_pylist():
        speed_kmh = float(speed_by_class.get(row["segment_class"], default_speed_kmh))
        if speed_kmh <= 0:
            continue
        length_m = float(row["length_m"]) if row["length_m"] and row["length_m"] > 0 else 0.1
        speed_mps = speed_kmh / 3.6
        weight_ds = max(1, round(length_m / speed_mps * 10))
        node_coords[row["from_connector"]] = (row["from_lat"], row["from_lng"])
        node_coords[row["to_connector"]] = (row["to_lat"], row["to_lng"])
        tail = (length_m, speed_kmh, weight_ds, row["segment_class"], row["segment_id"], row["name"])
        records.append((row["from_connector"], row["to_connector"], *tail))
        records.append((row["to_connector"], row["from_connector"], *tail))

    nodes_df, connector_to_idx = _nodes_table(node_coords)
    edges_df = _edges_table(records, connector_to_idx)
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
) -> tuple[pa.Table, pa.Table]:
    """Load a saved rail graph from disk, or fetch it from Overture Maps and save it.

    Arguments are those of ``fetch_rail_network`` plus the two output paths. If both files exist they are read
    (no download); otherwise the graph is fetched and written there as Parquet.

    Parameters
    ----------
    min_lon : float
        Bounding box in degrees (WGS 84).
    min_lat : float
        Bounding box in degrees (WGS 84).
    max_lon : float
        Bounding box in degrees (WGS 84).
    max_lat : float
        Bounding box in degrees (WGS 84).
    overture_release : str
        Overture Maps release id, for example ``"2026-10-21.0"``. Old releases are removed from the public bucket,
        so choose a current one.
    nodes_output : str
        Paths of the Parquet files for the graph nodes and edges.
    edges_output : str
        Paths of the Parquet files for the graph nodes and edges.
    classes : optional
        As in ``fetch_rail_network``.
    speed_kmh_by_class : optional
        As in ``fetch_rail_network``.
    default_speed_kmh : optional
        As in ``fetch_rail_network``.

    Returns
    -------
    (nodes, edges) : tuple of pyarrow.Table
        As returned by ``fetch_rail_network``.
    """
    if Path(nodes_output).exists() and Path(edges_output).exists():
        logger.info("Loading cached rail graph from %s / %s ...", nodes_output, edges_output)
        return pq.read_table(nodes_output), pq.read_table(edges_output)

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
    pq.write_table(nodes_df, nodes_output)
    pq.write_table(edges_df, edges_output)
    logger.info(
        "Saved rail graph: %d nodes, %d directed edges -> %s, %s",
        nodes_df.num_rows,
        edges_df.num_rows,
        nodes_output,
        edges_output,
    )
    return nodes_df, edges_df
