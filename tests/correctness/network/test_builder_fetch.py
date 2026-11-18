"""fetch_road_network / fetch_rail_network against a local Parquet file with the Overture segment schema."""

from __future__ import annotations

import pytest
from fastmob.network import builder

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def segments(tmp_path_factory):
    path = tmp_path_factory.mktemp("overture") / "segments.parquet"
    try:
        duckdb.sql("INSTALL spatial; LOAD spatial;")
    except Exception as error:  # noqa: BLE001 - no network to fetch the extension
        pytest.skip(f"DuckDB spatial extension unavailable: {error}")
    # id, subtype, class, name, speed limit (km/h), denied access, WKT (lon lat), connectors as (id, at)
    rows = [
        (
            "curve",
            "road",
            "residential",
            "Curve Street",
            None,
            None,
            "LINESTRING(0 0, 0.001 0.001, 0.002 0)",
            ["a", "b"],
        ),
        ("limit", "road", "primary", None, 50, None, "LINESTRING(0 0.01, 0.01 0.01)", ["c", "d"]),
        ("oneway", "road", "residential", "One Way", None, "forward", "LINESTRING(0 0.02, 0.01 0.02)", ["e", "f"]),
        ("footway", "road", "footway", "Park Path", None, None, "LINESTRING(0 0.03, 0.01 0.03)", ["g", "h"]),
        ("nofoot", "road", "service", "Private", None, "foot", "LINESTRING(0 0.04, 0.01 0.04)", ["i", "j"]),
        ("motorway", "road", "motorway", "Highway", None, None, "LINESTRING(0 0.05, 0.01 0.05)", ["k", "l"]),
        ("tram", "rail", "tram", "Line 1", None, None, "LINESTRING(0 0.06, 0.01 0.06)", ["m", "n"]),
    ]
    values = []
    for sid, subtype, cls, name, limit, denied, wkt, conns in rows:
        limits = f"[{{'max_speed': {{'value': {limit}, 'unit': 'km/h'}}, 'between': NULL}}]" if limit else "NULL"
        if denied == "forward":
            access = "[{'access_type': 'denied', 'when': {'heading': 'forward', 'mode': NULL}}]"
        elif denied == "foot":
            access = "[{'access_type': 'denied', 'when': {'heading': NULL, 'mode': ['foot']}}]"
        else:
            access = "NULL"
        name_sql = f"'{name}'" if name else "NULL"
        connectors = f"[{{'connector_id': '{conns[0]}', 'at': 0.0::DOUBLE}}, {{'connector_id': '{conns[1]}', 'at': 1.0::DOUBLE}}]"
        values.append(
            f"('{sid}', '{subtype}', '{cls}', {{'primary': {name_sql}}}, {limits}, {access}, "
            f"ST_GeomFromText('{wkt}'), {connectors}, {{'xmin': 0.0::DOUBLE, 'ymin': 0.0::DOUBLE}})"
        )
    duckdb.sql(
        "CREATE OR REPLACE TABLE seg AS SELECT * FROM (VALUES "
        + ", ".join(values)
        + ") t(id, subtype, class, names, speed_limits, access_restrictions, geometry, connectors, bbox)"
    )
    duckdb.sql(f"COPY seg TO '{path}' (FORMAT parquet)")
    return str(path)


@pytest.fixture
def local_source(segments, monkeypatch):
    monkeypatch.setattr(builder, "_segment_source", lambda release: segments)


BOX = (-1.0, -1.0, 1.0, 1.0)


def test_drive_network_keeps_street_identity_and_directions(local_source):
    nodes, edges = builder.fetch_road_network(*BOX, "test")
    rows = edges.to_pylist()
    assert {"segment_id", "name"} <= set(edges.column_names)
    by_segment = {}
    for row in rows:
        by_segment.setdefault(row["segment_id"], []).append(row)
    assert set(by_segment) == {"curve", "limit", "oneway", "motorway", "nofoot"}  # footways are not drivable
    assert len(by_segment["curve"]) == 2 and len(by_segment["oneway"]) == 1
    assert {r["name"] for r in by_segment["curve"]} == {"Curve Street"}
    assert by_segment["limit"][0]["name"] is None
    assert by_segment["limit"][0]["speed_kmh"] == 50.0
    assert nodes.num_rows == 10


def test_length_follows_the_segment_not_the_chord(local_source):
    _, edges = builder.fetch_road_network(*BOX, "test")
    curve = next(r for r in edges.to_pylist() if r["segment_id"] == "curve")
    chord = 0.002 * 111_195  # the straight line between the two connectors, about 222 m
    assert curve["length_m"] > 1.3 * chord  # the path goes up and down, about 314 m


def test_walk_network_is_undirected_with_walking_speed_and_footways(local_source):
    _, edges = builder.fetch_road_network(*BOX, "test", mode="walk")
    by_segment = {}
    for row in edges.to_pylist():
        by_segment.setdefault(row["segment_id"], []).append(row)
    assert set(by_segment) == {"curve", "limit", "oneway", "footway"}  # nofoot is denied, motorway is not walkable
    assert all(len(rows) == 2 for rows in by_segment.values())  # every piece both ways, even the one-way street
    assert {r["speed_kmh"] for rows in by_segment.values() for r in rows} == {5.04}
    footway = by_segment["footway"][0]
    assert footway["weight_ds"] == round(footway["length_m"] / 1.4 * 10)


def test_walk_options_override_classes_speed_and_direction(local_source):
    _, edges = builder.fetch_road_network(
        *BOX, "test", mode="walk", classes=["footway", "residential"], speed_kmh_by_class={"footway": 4.0}
    )
    speeds = {r["segment_id"]: r["speed_kmh"] for r in edges.to_pylist()}
    assert speeds == {"footway": 4.0, "curve": 5.04, "oneway": 5.04}
    _, undirected = builder.fetch_road_network(*BOX, "test", directed=False)
    assert sum(r["segment_id"] == "oneway" for r in undirected.to_pylist()) == 2


def test_rail_network_keeps_segment_id_and_name(local_source):
    _, edges = builder.fetch_rail_network(*BOX, "test")
    assert {(r["segment_id"], r["name"]) for r in edges.to_pylist()} == {("tram", "Line 1")}


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        builder.fetch_road_network(*BOX, "test", mode="bike")
