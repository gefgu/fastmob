from __future__ import annotations

from pathlib import Path

from benchmarks.preprocessing import speed_suite as suite


def test_preprocessing_registry_contains_expected_functions():
    names = [spec.name for spec in suite.PREPROCESSING_METRICS]

    assert names == ["filter", "compress", "stay_locations", "cluster"]


def test_output_path_uses_preprocessing_suite_name(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "pandas")
        == tmp_path / "skmob2_preprocessing_speed_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_preprocessing_speed_prebuilt_tdf.json"
    )
