from __future__ import annotations

from benchmarks import integration_poi_event_speed as suite


def test_build_synthetic_data_produces_expected_sizes():
    import numpy as np

    rng = np.random.default_rng(0)
    traj_df, pois_df, events_df = suite._build_synthetic_data(50, rng)
    assert len(traj_df) == 50
    assert len(pois_df) == suite.N_POIS
    assert len(events_df) == suite.N_EVENTS
    assert set(traj_df.columns) >= {"lat", "lng", "datetime"}
    assert set(pois_df.columns) >= {"lat", "lng", "id", "name_poi", "type_poi"}
    assert set(events_df.columns) >= {"lat", "lng", "datetime", "event_id", "event_type"}


def test_main_runs_at_tiny_scale(tmp_path):
    output_path = tmp_path / "result.json"
    exit_code = suite.main(
        [
            "--traj-sizes",
            "10",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert output_path.exists()

    import json

    payload = json.loads(output_path.read_text())
    by_size = payload["results"]["by_traj_size"]
    assert len(by_size) == 1
    assert by_size[0]["n_traj"] == 10
