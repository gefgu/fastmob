from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.shared.geolife import load_geolife_pandas, read_geolife_plt_text

PLT_TEXT = """Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,Geolife Track
0
39.906631,116.385564,0,492,40097.5864583333,2009-10-11,14:04:30
39.906554,116.385625,0,492,40097.5865162037,2009-10-11,14:04:35
"""


def test_read_geolife_plt_text_skips_header_and_builds_timestamp():
    df = read_geolife_plt_text(PLT_TEXT, user="123", trajectory_id="20091011140430")

    assert list(df.columns) == ["user", "check-in_time", "latitude", "longitude", "trajectory_id"]
    assert df["user"].tolist() == ["123", "123"]
    assert df["trajectory_id"].tolist() == ["123/20091011140430", "123/20091011140430"]
    assert df["check-in_time"].tolist() == [
        pd.Timestamp("2009-10-11 14:04:30"),
        pd.Timestamp("2009-10-11 14:04:35"),
    ]
    assert df["latitude"].tolist() == [39.906631, 39.906554]
    assert df["longitude"].tolist() == [116.385564, 116.385625]


def test_load_geolife_slice_is_deterministic_and_row_limited(tmp_path: Path):
    data_root = tmp_path / "Geolife Trajectories 1.3" / "Data"
    for user, trajectory, lat in [
        ("001", "b", 41.0),
        ("000", "b", 40.0),
        ("000", "a", 39.0),
    ]:
        trajectory_dir = data_root / user / "Trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        (trajectory_dir / f"{trajectory}.plt").write_text(PLT_TEXT.replace("39.906631", str(lat)), encoding="utf-8")

    df = load_geolife_pandas(mode="slice", rows=3, dataset_root=tmp_path)

    assert len(df) == 3
    assert df["user"].tolist() == ["000", "000", "000"]
    assert df["trajectory_id"].tolist() == ["000/a", "000/b", "000/a"]
    assert df["check-in_time"].is_monotonic_increasing
