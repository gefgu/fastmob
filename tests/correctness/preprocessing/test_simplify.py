"""Correctness tests for fastmob.preprocessing.simplify.

Each hand-crafted fixture in ``conftest.py`` (``simplify_tdf`` /
``simplify_tdf_polars``) isolates one documented behavior per algorithm; see
the fixture docstring comments there for the exact geometry/timing choices.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.preprocessing import simplify


def _user(df, uid: str):
    if hasattr(df, "loc"):
        return df[df["uid"] == uid].reset_index(drop=True)
    import polars as pl

    return df.filter(pl.col("uid") == uid)


def test_douglas_peucker_drops_near_collinear_middle_point(simplify_tdf):
    """3 near-collinear points (~0.7 m lateral offset) within a 10 m epsilon
    → the middle point is dropped."""
    user = _user(simplify_tdf, "near_collinear")
    result = simplify(user, method="douglas_peucker", epsilon_km=0.01)
    assert len(result) == 2
    assert sorted(result["lat"].to_list()) == [48.8560, 48.8570]


def test_top_down_time_ratio_keeps_spatially_collinear_but_early_point(simplify_tdf):
    """Middle point is spatially the exact midpoint of the endpoints, but
    arrives after only 1% of the elapsed time between them: Douglas-Peucker
    (spatial-only) drops it, top_down_time_ratio (spatiotemporal) keeps it."""
    user = _user(simplify_tdf, "time_ratio")

    dp_result = simplify(user, method="douglas_peucker", epsilon_km=0.01)
    tdtr_result = simplify(user, method="top_down_time_ratio", epsilon_km=0.01)

    assert len(dp_result) == 2
    assert len(tdtr_result) == 3


def test_min_distance_drops_point_closer_than_threshold(simplify_tdf):
    """Middle point ~5.5 m from the first kept point (below a 20 m
    threshold) is dropped; last point ~33 m away (above it) is kept."""
    user = _user(simplify_tdf, "min_dist")
    result = simplify(user, method="min_distance", min_distance_km=0.02)
    assert len(result) == 2
    assert result["lat"].to_list()[0] == pytest.approx(48.8560)
    assert result["lat"].to_list()[1] == pytest.approx(48.8560 + 0.0003)


def test_min_time_delta_drops_point_sooner_than_threshold(simplify_tdf):
    """Middle point arrives 10s after the first kept point (below a 60s
    threshold) and is dropped; last point arrives 120s after (above it) and
    is kept."""
    user = _user(simplify_tdf, "min_td")
    result = simplify(user, method="min_time_delta", min_time_delta_s=60.0)
    assert len(result) == 2
    assert result["lat"].to_list() == [48.8560, 48.8562]


def test_max_distance_drops_near_collinear_middle_point(simplify_tdf):
    """Like Douglas-Peucker, MaxDistance's single-pass streaming check also
    drops a near-collinear middle point within epsilon."""
    user = _user(simplify_tdf, "near_collinear")
    result = simplify(user, method="max_distance", epsilon_km=0.01)
    assert len(result) == 2
    assert sorted(result["lat"].to_list()) == [48.8560, 48.8570]


def test_chan_chin_keeps_corner_vertex(simplify_tdf):
    """A 90-degree turn's corner point must be kept: dropping it would
    create a ~1 km deviation, far above any reasonable epsilon."""
    user = _user(simplify_tdf, "corner")
    result = simplify(user, method="chan_chin", epsilon_km=0.01)
    assert len(result) == 3


def test_chan_chin_drops_near_collinear_middle_point(simplify_tdf):
    """Near-collinear middle point within epsilon is dropped."""
    user = _user(simplify_tdf, "near_collinear")
    result = simplify(user, method="chan_chin", epsilon_km=0.01)
    assert len(result) == 2


def test_imai_iri_keeps_corner_vertex(simplify_tdf):
    """A 90-degree turn's corner point must be kept: dropping it would
    create a ~1 km deviation, far above any reasonable epsilon."""
    user = _user(simplify_tdf, "corner")
    result = simplify(user, method="imai_iri", epsilon_km=0.01)
    assert len(result) == 3


def test_imai_iri_drops_near_collinear_middle_point(simplify_tdf):
    """Near-collinear middle point within epsilon is dropped."""
    user = _user(simplify_tdf, "near_collinear")
    result = simplify(user, method="imai_iri", epsilon_km=0.01)
    assert len(result) == 2


def test_simplify_single_point_user():
    """Single-point user passes through unchanged (nothing to simplify)."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = simplify(df, method="douglas_peucker", epsilon_km=0.01)
    assert len(result) == 1


def test_simplify_two_point_user():
    """Two-point user always keeps both points (both endpoints, nothing to
    simplify between them)."""
    df = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-01 00:01:00"),
            ],
            "lat": [48.8566, 48.8567],
            "lng": [2.3522, 2.3522],
        }
    )
    result = simplify(df, method="chan_chin", epsilon_km=0.001)
    assert len(result) == 2


def test_simplify_multiuser_all_users_processed(simplify_tdf):
    """Multi-user input: every user is processed independently."""
    result = simplify(simplify_tdf, method="douglas_peucker", epsilon_km=0.01)
    assert set(result["uid"].unique()) == set(simplify_tdf["uid"].unique())


def test_simplify_returns_same_backend_type(simplify_tdf):
    """Pandas input → pandas output."""
    result = simplify(simplify_tdf, method="douglas_peucker", epsilon_km=0.01)
    assert isinstance(result, type(simplify_tdf))


def test_simplify_polars_backend(simplify_tdf_polars):
    """Polars input → Polars output with the expected row count."""
    import polars as pl

    result = simplify(simplify_tdf_polars, method="douglas_peucker", epsilon_km=0.01)
    assert isinstance(result, pl.DataFrame)
    near_collinear = result.filter(pl.col("uid") == "near_collinear")
    assert len(near_collinear) == 2


def test_simplify_unknown_method_raises(simplify_tdf):
    """An unrecognized ``method`` raises ValueError listing valid choices."""
    with pytest.raises(ValueError, match="unknown simplify method"):
        simplify(simplify_tdf, method="not_a_real_method")


def _per_user_jaccard(result_df, cached_kept_by_uid: dict) -> dict:
    """Return per-uid Jaccard similarity between fastmob's and the cached kept-row-index sets."""
    scores = {}
    for uid, group in result_df.groupby("uid"):
        fastmob_set = set(group["row_index"].tolist())
        cached_set = cached_kept_by_uid.get(uid, set())
        union = fastmob_set | cached_set
        scores[uid] = len(fastmob_set & cached_set) / len(union) if union else 1.0
    return scores


@pytest.mark.parametrize(
    ("method", "kwargs", "min_mean_jaccard"),
    [
        ("min_distance", {"min_distance_km": 0.2}, 1.0),
        ("min_time_delta", {"min_time_delta_s": 600.0}, 1.0),
        ("max_distance", {"epsilon_km": 0.05}, 1.0),
        ("top_down_time_ratio", {"epsilon_km": 0.05}, 0.95),
        ("douglas_peucker", {"epsilon_km": 0.05}, 0.6),
    ],
)
def test_simplify_matches_cached_movingpandas_reference(movingpandas_reference, method, kwargs, min_mean_jaccard):
    """Row-subset agreement with the cached MovingPandas baseline on a Brightkite slice.

    ``min_distance``/``min_time_delta``/``max_distance`` are direct ports of
    MovingPandas' reference algorithms and match exactly (Jaccard == 1.0 per
    user in the cached run). ``top_down_time_ratio`` is a faithful iterative
    port and matches almost exactly. ``douglas_peucker`` is compared with a
    wider, documented tolerance: MovingPandas runs GEOS' Douglas-Peucker on
    each user's estimated-UTM-projected coordinates, while fastmob runs
    geo-rust's Douglas-Peucker on its own local equirectangular planar-km
    projection — two independent RDP implementations on two different (but
    both locally accurate) projections, whose farthest-point tie-breaks can
    diverge through the recursive splitting even though both stay within
    epsilon of the original trajectory.
    """
    input_df = movingpandas_reference.input_df
    cached = movingpandas_reference.kept_row_index(method)
    if cached is None:
        pytest.skip(f"No cached MovingPandas result for method={method!r}")

    result = simplify(
        input_df,
        method=method,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        **kwargs,
    )
    scores = _per_user_jaccard(result, cached)
    mean_score = sum(scores.values()) / len(scores)
    assert mean_score >= min_mean_jaccard, scores


@pytest.mark.parametrize("method", ["chan_chin", "imai_iri"])
def test_simplify_matches_cached_movetk_reference(movetk_reference, method):
    """Row-subset agreement with the cached MoveTK baseline on the same Brightkite slice.

    The cache was populated by building and running MoveTK's real
    ``ChanChin``/``ImaiIri`` classes (see
    ``tests/shared/movetk_cache.py``'s module docstring) on the same
    coordinates fastmob's own ``chan_chin``/``imai_iri`` project to
    (removing projection differences as a variable).

    This is a wide, documented tolerance rather than a strict match: MoveTK's
    ``Wedge`` is a tangent-line-to-a-circle construction with its own
    "degenerate" branch for points already within epsilon of the anchor,
    while fastmob's corridor primitive uses a simpler arc/angle feasibility
    test that treats such points as imposing no constraint at all. On this
    Brightkite slice, that difference makes fastmob noticeably more
    conservative (it keeps a superset of MoveTK's kept points on every
    user), so this test tracks gross regressions rather than asserting tight
    numeric parity. Closing that gap is a documented follow-up, not required
    for this phase.
    """
    from tests.shared.movingpandas_cache import MovingPandasReferenceDataset

    cached = movetk_reference.kept_row_index(method)
    if cached is None:
        pytest.skip(f"No cached MoveTK result for method={method!r}")

    input_df = MovingPandasReferenceDataset("brightkite").input_df
    result = simplify(
        input_df,
        method=method,
        uid_col="uid",
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        epsilon_km=0.05,
    )
    # JSON object keys are always strings; input_df's uid column is int64.
    cached_by_uid = {int(uid): set(v) for uid, v in cached.items()}
    scores = _per_user_jaccard(result, cached_by_uid)
    mean_score = sum(scores.values()) / len(scores)
    assert mean_score >= 0.5, scores


@pytest.mark.skip(reason="agarwal simplification deferred, see plan doc")
def test_agarwal_simplification_placeholder():
    """Placeholder for the deferred Agarwal simplification algorithm.

    Agarwal et al.'s simplification finds a near-optimal minimum-error
    simplification using a parametric-search feasibility region that is
    materially more complex than the Wedge/corridor primitive built for
    Chan-Chin and Imai-Iri, with limited value-add over Chan-Chin's tighter
    approximation bound alone (see the project plan's "What ships vs. what's
    deferred" table). When implemented, this test should assert:

    - ``fastmob.preprocessing.simplify(traj, method="agarwal", epsilon_km=...)``
      returns a simplification whose worst-case point-to-segment error is
      bounded by ``epsilon_km``, with an approximation ratio at least as
      tight as Chan-Chin's for the same tolerance.
    - On a hand-crafted near-collinear fixture (e.g. this file's
      ``near_collinear`` case), the redundant middle point is dropped.
    - On a hand-crafted corner fixture (e.g. this file's ``corner`` case),
      the corner vertex is always kept.
    """
    raise NotImplementedError
