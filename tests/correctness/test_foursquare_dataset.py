from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.shared.foursquare import FOURSQUARE_FILE_NAME, load_foursquare_pandas, read_foursquare_text

FOURSQUARE_TEXT = "1\tvenue_b\tcat_b\tCoffee Shop\t40.7001\t-73.9001\t-240\tTue Apr 03 18:00:09 +0000 2012\n1\tvenue_a\tcat_a\tOffice\t40.7000\t-73.9000\t-240\tTue Apr 03 18:00:09 +0000 2012\n2\tvenue_c\tcat_c\tPark\t40.8000\t-73.8000\t-240\tWed Apr 04 19:30:00 +0000 2012"


def test_read_foursquare_text_builds_normalized_columns():
    df = read_foursquare_text(FOURSQUARE_TEXT)

    assert list(df.columns) == ["user", "check-in_time", "latitude", "longitude", "location_id"]
    assert df["user"].tolist() == ["1", "1", "2"]
    assert df["location_id"].tolist() == ["venue_b", "venue_a", "venue_c"]
    assert df["check-in_time"].tolist() == [
        pd.Timestamp("2012-04-03 18:00:09"),
        pd.Timestamp("2012-04-03 18:00:09"),
        pd.Timestamp("2012-04-04 19:30:00"),
    ]
    assert df["latitude"].tolist() == [40.7001, 40.7000, 40.8000]
    assert df["longitude"].tolist() == [-73.9001, -73.9000, -73.8000]


def test_load_foursquare_slice_is_deterministic_row_limited_and_deduplicated(tmp_path: Path):
    data_dir = tmp_path / "dataset_tsmc2014"
    data_dir.mkdir()
    (data_dir / FOURSQUARE_FILE_NAME).write_text(
        "2\tvenue_c\tcat_c\tPark\t40.8000\t-73.8000\t-240\tWed Apr 04 19:30:00 +0000 2012\n1\tvenue_b\tcat_b\tCoffee Shop\t40.7001\t-73.9001\t-240\tTue Apr 03 18:00:09 +0000 2012\n1\tvenue_a\tcat_a\tOffice\t40.7000\t-73.9000\t-240\tTue Apr 03 18:00:09 +0000 2012\n1\tvenue_d\tcat_d\tGym\t40.9000\t-73.7000\t-240\tThu Apr 05 20:00:00 +0000 2012",
        encoding="utf-8",
    )

    df = load_foursquare_pandas(mode="slice", rows=2, dataset_root=tmp_path)

    assert len(df) == 2
    assert df["user"].tolist() == ["1", "1"]
    assert df["location_id"].tolist() == ["venue_a", "venue_d"]
    assert df["check-in_time"].is_monotonic_increasing
