"""fetch_road_network / fetch_rail_network against a local Parquet file with the Overture segment schema."""

from __future__ import annotations

import pytest
from fastmob.network import builder

duckdb = pytest.importorskip("duckdb")


def _entry(access_type, heading=None, mode=None, using=None):
    """One Overture access restriction as a DuckDB struct literal with fixed field types."""
    heading_sql = f"'{heading}'::VARCHAR" if heading else "NULL::VARCHAR"
    mode_sql = f"{mode}::VARCHAR[]" if mode else "NULL::VARCHAR[]"
    using_sql = f"{using}::VARCHAR[]" if using else "NULL::VARCHAR[]"
    return f"{{'access_type': '{access_type}', 'when': {{'heading': {heading_sql}, 'mode': {mode_sql}, 'using': {using_sql}}}}}"


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
        ("pedonly", "road", "service", "Plaza", None, "pedestrian_only", "LINESTRING(0 0.07, 0.01 0.07)", ["o", "p"]),
        ("customers", "road", "service", "Shop Lane", None, "customers", "LINESTRING(0 0.08, 0.01 0.08)", ["q", "r"]),
        ("footoneway", "road", "footway", "Stair", None, "foot_forward", "LINESTRING(0 0.09, 0.01 0.09)", ["s", "t"]),
        ("north", "road", "residential", "Northern", None, None, "LINESTRING(0 60, 0.01 60)", ["u", "v"]),
        ("outside", "road", "residential", "Elsewhere", None, None, "LINESTRING(5 5, 5.01 5)", ["w", "x"]),
        ("tram", "rail", "tram", "Line 1", None, None, "LINESTRING(0 0.06, 0.01 0.06)", ["m", "n"]),
    ]
    values = []
    for sid, subtype, cls, name, limit, denied, wkt, conns in rows:
        limits = f"[{{'max_speed': {{'value': {limit}, 'unit': 'km/h'}}, 'between': NULL}}]" if limit else "NULL"
        access = {
            None: "NULL",
            "forward": f"[{_entry('denied', heading='forward')}]",
            "foot": f"[{_entry('denied', mode=['foot'])}]",
            "pedestrian_only": f"[{_entry('denied')}, {_entry('allowed', mode=['foot'])}]",
            "customers": f"[{_entry('denied', using=['as_customer'])}]",
            "foot_forward": f"[{_entry('denied', heading='forward', mode=['foot'])}]",
        }[denied]
        name_sql = f"'{name}'" if name else "NULL"
        connectors = f"[{{'connector_id': '{conns[0]}', 'at': 0.0::DOUBLE}}, {{'connector_id': '{conns[1]}', 'at': 1.0::DOUBLE}}]"
        xmin, ymin = wkt[wkt.index("(") + 1 :].split(",")[0].split()
        values.append(
            f"('{sid}', '{subtype}', '{cls}', {{'primary': {name_sql}}}, {limits}, {access}, "
            f"ST_GeomFromText('{wkt}'), {connectors}, {{'xmin': {xmin}::DOUBLE, 'ymin': {ymin}::DOUBLE}})"
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
NORTH = (-1.0, 59.0, 1.0, 61.0)


def test_drive_network_keeps_street_identity_and_directions(local_source):
    nodes, edges = builder.fetch_road_network(*BOX, "test")
    rows = edges.to_pylist()
    assert {"segment_id", "name"} <= set(edges.column_names)
    by_segment = {}
    for row in rows:
        by_segment.setdefault(row["segment_id"], []).append(row)
    assert set(by_segment) == {
        "curve",
        "limit",
        "oneway",
        "motorway",
        "nofoot",
    }  # footways and pedestrian-only streets are not drivable
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
    # nofoot is denied; the motorway is not walkable; a pedestrian-only street is allowed to foot; a restriction
    # that only applies to customers does not close the lane
    assert set(by_segment) == {"curve", "limit", "oneway", "footway", "pedonly", "customers", "footoneway"}
    assert all(len(rows) == 2 for rows in by_segment.values())  # every piece both ways, even the one-way street
    assert {r["speed_kmh"] for rows in by_segment.values() for r in rows} == {5.04}
    footway = by_segment["footway"][0]
    assert footway["weight_ds"] == round(footway["length_m"] / 1.4 * 10)


def test_walk_options_override_classes_speed_and_direction(local_source):
    _, edges = builder.fetch_road_network(
        *BOX, "test", mode="walk", classes=["footway", "residential"], speed_kmh_by_class={"footway": 4.0}
    )
    speeds = {r["segment_id"]: r["speed_kmh"] for r in edges.to_pylist()}
    assert speeds == {"footway": 4.0, "footoneway": 4.0, "curve": 5.04, "oneway": 5.04}
    _, undirected = builder.fetch_road_network(*BOX, "test", directed=False)
    assert sum(r["segment_id"] == "oneway" for r in undirected.to_pylist()) == 2


def test_walk_directed_honours_foot_headings_only(local_source):
    _, edges = builder.fetch_road_network(*BOX, "test", mode="walk", directed=True)
    count = {}
    for row in edges.to_pylist():
        count[row["segment_id"]] = count.get(row["segment_id"], 0) + 1
    assert count["footoneway"] == 1  # a foot restriction in one heading
    assert count["oneway"] == 1  # a restriction that names no mode applies to everyone, pedestrians included


def test_length_is_measured_in_the_right_axis_order_away_from_the_equator(local_source):
    _, edges = builder.fetch_road_network(*NORTH, "test")
    north = next(r for r in edges.to_pylist() if r["segment_id"] == "north")
    # 0.01 degrees of longitude at 60 N is 0.01 * 111_195 * cos(60 deg) = 556 m; with swapped axes it would be 1,112 m
    assert north["length_m"] == pytest.approx(556, rel=0.01)


def test_segments_outside_the_box_and_empty_class_lists(local_source):
    _, edges = builder.fetch_road_network(*BOX, "test")
    assert "outside" not in {r["segment_id"] for r in edges.to_pylist()}
    with pytest.raises(ValueError, match="at least one road class"):
        builder.fetch_road_network(*BOX, "test", classes=[])
    _, edges = builder.fetch_road_network(*BOX, "test", classes=["residential'; --"])
    assert edges.num_rows == 0


def test_rail_network_keeps_segment_id_and_name(local_source):
    _, edges = builder.fetch_rail_network(*BOX, "test")
    assert {(r["segment_id"], r["name"]) for r in edges.to_pylist()} == {("tram", "Line 1")}


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        builder.fetch_road_network(*BOX, "test", mode="bike")
