"""Correctness tests for fastmob.io.file (pyarrow-backed CSV/parquet I/O)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from fastmob.core import TrajDataFrame
from fastmob.io.file import load_geolife_trajectories, read, write


def _sample_tdf() -> TrajDataFrame:
    return TrajDataFrame(
        pa.table(
            {
                "uid": [1, 1, 2],
                "lat": [40.0, 40.1, 41.0],
                "lng": [-73.0, -73.1, -74.0],
                "datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
            }
        )
    )


def test_write_then_read_csv_round_trip(tmp_path: Path):
    tdf = _sample_tdf()
    out = tmp_path / "trajectory.csv"
    write(tdf, out)
    assert out.exists()

    loaded = read(out)
    assert isinstance(loaded, TrajDataFrame)
    assert sorted(loaded.df.column_names) == sorted(tdf.df.column_names)
    assert loaded.df.num_rows == tdf.df.num_rows


def test_write_then_read_parquet_round_trip(tmp_path: Path):
    tdf = _sample_tdf()
    out = tmp_path / "trajectory.parquet"
    write(tdf, out)
    assert out.exists()

    loaded = read(out)
    assert isinstance(loaded, TrajDataFrame)
    assert loaded.df.num_rows == tdf.df.num_rows


def test_write_accepts_polars_backed_trajdataframe(tmp_path: Path):
    """Regression test: write() previously called .to_csv/.to_parquet
    directly on tdf.df, which raises AttributeError on modern polars
    (only .write_csv/.write_parquet exist there). Normalizing through
    Narwhals -> pyarrow fixes this for every backend uniformly."""
    pl = pytest.importorskip("polars", reason="Polars not installed")
    tdf = TrajDataFrame(
        pl.DataFrame(
            {
                "uid": [1, 1],
                "lat": [40.0, 40.1],
                "lng": [-73.0, -73.1],
                "datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            }
        )
    )
    out = tmp_path / "polars_trajectory.csv"
    write(tdf, out)
    assert out.exists()
    loaded = read(out)
    assert loaded.df.num_rows == 2


def test_read_csv_passes_kwargs_to_arrow_reader(tmp_path: Path):
    out = tmp_path / "semicolon.csv"
    out.write_text("uid;lat;lng;datetime\n1;40.0;-73.0;2020-01-01 00:00:00\n", encoding="utf-8")
    loaded = read(out, delimiter=";")
    assert loaded.df.num_rows == 1


_PLT_TEXT = """Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,Geolife Track
0
39.906631,116.385564,0,492,40097.5864583333,2009-10-11,14:04:30
39.906554,116.385625,0,492,40097.5865162037,2009-10-11,14:04:35
"""


def test_load_geolife_trajectories_parses_plt_files(tmp_path: Path):
    trajectory_dir = tmp_path / "Data" / "123" / "Trajectory"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "20091011140430.plt").write_text(_PLT_TEXT, encoding="utf-8")

    tdf = load_geolife_trajectories(tmp_path)
    assert isinstance(tdf, TrajDataFrame)
    assert tdf.df.num_rows == 2
    assert tdf.df["uid"].to_pylist() == ["123", "123"]
    assert tdf.df["lat"].to_pylist() == pytest.approx([39.906631, 39.906554])


def test_load_geolife_trajectories_filters_by_user_ids(tmp_path: Path):
    for user in ("100", "200"):
        trajectory_dir = tmp_path / "Data" / user / "Trajectory"
        trajectory_dir.mkdir(parents=True)
        (trajectory_dir / "traj.plt").write_text(_PLT_TEXT, encoding="utf-8")

    tdf = load_geolife_trajectories(tmp_path, user_ids=["100"])
    assert set(tdf.df["uid"].to_pylist()) == {"100"}
