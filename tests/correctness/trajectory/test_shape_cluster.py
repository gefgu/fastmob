"""Correctness tests for fastmob.trajectory.cluster_trajectory_shapes /
cluster_trajectory_shapes_from_segments.

Uses hand-built families of trajectories with obviously matching or
distinct shapes: straight lines of different length/position/rotation
should cluster together; an L-shaped path should be distinct/noise. Also
checks the rotation/translation/scale-invariance property directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.trajectory import cluster_trajectory_shapes, cluster_trajectory_shapes_from_segments


def _line(lat0: float, lng0: float, n: int = 5, step: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [lat0 + i * step for i in range(n)],
            "lng": [lng0] * n,
            "datetime": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )


def _rotated_line(lat0: float, lng0: float, n: int, step: float, rotate_deg: float) -> pd.DataFrame:
    lats = np.array([lat0 + i * step for i in range(n)], dtype=float)
    lngs = np.array([lng0] * n, dtype=float)
    theta = np.radians(rotate_deg)
    dlat, dlng = lats - lats[0], lngs - lngs[0]
    rlat = lats[0] + dlat * np.cos(theta) - dlng * np.sin(theta)
    rlng = lngs[0] + dlat * np.sin(theta) + dlng * np.cos(theta)
    return pd.DataFrame({"lat": rlat, "lng": rlng, "datetime": pd.date_range("2020-01-01", periods=n, freq="h")})


def _l_shape() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [0.0, 1.0, 1.0, 1.0],
            "lng": [0.0, 0.0, 1.0, 2.0],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
        }
    )


def _circle(lat0: float, lng0: float, n: int = 8, radius: float = 0.01) -> pd.DataFrame:
    angles = np.linspace(0, 2 * np.pi, n)
    return pd.DataFrame(
        {
            "lat": lat0 + radius * np.sin(angles),
            "lng": lng0 + radius * np.cos(angles),
            "datetime": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )


def test_straight_lines_at_different_positions_and_lengths_cluster_together():
    trajs = [_line(0, 0), _line(10, 10), _line(20, 20, step=2.0)]
    labels = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.1, min_cluster_size=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[0] != -1


def test_lines_and_circles_form_distinct_clusters():
    trajs = [_line(0, 0), _line(10, 10), _line(20, 20, step=2.0), _circle(0, 0), _circle(5, 5)]
    labels = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.1, min_cluster_size=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_rotation_invariance_clusters_rotated_line_with_original():
    trajs = [_line(0, 0), _rotated_line(50, 50, n=5, step=1.0, rotate_deg=45.0)]
    labels = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.15, min_cluster_size=2)
    assert labels[0] == labels[1]
    assert labels[0] != -1


def test_l_shape_is_distinct_from_straight_lines():
    trajs = [_line(0, 0), _line(10, 10), _l_shape()]
    labels = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.1, min_cluster_size=2)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_deterministic_given_fixed_config():
    trajs = [_line(0, 0), _line(10, 10), _circle(0, 0), _circle(5, 5)]
    labels_1 = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.1, min_cluster_size=2)
    labels_2 = cluster_trajectory_shapes(trajs, depth=3, epsilon=0.1, min_cluster_size=2)
    assert labels_1.tolist() == labels_2.tolist()


def test_empty_list_returns_empty_array():
    labels = cluster_trajectory_shapes([], depth=2)
    assert len(labels) == 0


def test_depth_must_be_positive():
    with pytest.raises(ValueError, match="depth must be >= 1"):
        cluster_trajectory_shapes([_line(0, 0)], depth=0)


def test_multi_user_element_raises():
    multi = pd.concat([_line(0, 0).assign(uid="u1"), _line(5, 5).assign(uid="u2")], ignore_index=True)
    with pytest.raises(ValueError, match="single user's trajectory"):
        cluster_trajectory_shapes([multi], depth=2)


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    trajs_pd = [_line(0, 0), _line(10, 10), _circle(0, 0), _circle(5, 5)]
    trajs_pl = [pl.from_pandas(t) for t in trajs_pd]

    labels_pd = cluster_trajectory_shapes(trajs_pd, depth=3, epsilon=0.1, min_cluster_size=2)
    labels_pl = cluster_trajectory_shapes(trajs_pl, depth=3, epsilon=0.1, min_cluster_size=2)
    assert labels_pl.tolist() == labels_pd.tolist()


def test_cluster_trajectory_shapes_from_segments():
    df = pd.concat(
        [
            _line(0, 0).assign(uid="u1", segment_id=0),
            _line(10, 10).assign(uid="u1", segment_id=1),
            _l_shape().assign(uid="u2", segment_id=0),
        ],
        ignore_index=True,
    )
    result = cluster_trajectory_shapes_from_segments(df, uid_col="uid", depth=3, epsilon=0.1, min_cluster_size=2)
    result = result.sort_values(["uid", "segment_id"]).reset_index(drop=True)
    assert len(result) == 3
    assert result.loc[0, "shape_cluster"] == result.loc[1, "shape_cluster"]
    assert result.loc[2, "shape_cluster"] != result.loc[0, "shape_cluster"]


# ---------------------------------------------------------------------------
# Real dataset: structural invariants only (no numeric ground truth for real
# trajectory-shape clustering).
# ---------------------------------------------------------------------------


def test_geolife_per_trajectory_clustering_structural_invariants():
    geolife = pytest.importorskip("tests.shared.geolife", reason="GeoLife loader unavailable")
    try:
        df = geolife.load_geolife_pandas(rows=20_000)
    except Exception as exc:  # noqa: BLE001 - dataset may not be downloadable in this environment
        pytest.skip(f"GeoLife dataset unavailable: {exc}")

    if "trajectory_id" not in df.columns:
        pytest.skip("GeoLife loader did not produce a trajectory_id column")

    trajs = []
    for _, group in df.groupby("trajectory_id"):
        group = group.rename(columns={"check-in_time": "datetime", "latitude": "lat", "longitude": "lng"})
        if len(group) >= 2:
            trajs.append(group.sort_values("datetime").reset_index(drop=True))
        if len(trajs) >= 30:
            break

    if len(trajs) < 3:
        pytest.skip("Not enough GeoLife trajectories with >=2 points available")

    labels = cluster_trajectory_shapes(trajs, depth=4, epsilon=0.1, min_cluster_size=2)
    assert len(labels) == len(trajs)

    labels_again = cluster_trajectory_shapes(trajs, depth=4, epsilon=0.1, min_cluster_size=2)
    assert labels.tolist() == labels_again.tolist()

    noise_ratio = float((labels == -1).mean())
    assert 0.0 <= noise_ratio <= 1.0
