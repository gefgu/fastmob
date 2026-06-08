from __future__ import annotations

import sys
import importlib

import numpy as np
import pandas as pd
import pytest

from skmob2.models import (
    DensityEPR,
    EPR,
    GeoSim,
    Gravity,
    MarkovDiaryGenerator,
    Radiation,
    STS_epr,
    SpatialEPR,
    exponential_deterrence_func,
    powerlaw_deterrence_func,
)


def _tessellation(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tile_id": list(range(n)),
            "lat": [45.0 + i * 0.03 for i in range(n)],
            "lng": [7.0 + i * 0.04 for i in range(n)],
            "relevance": [5 + i for i in range(n)],
            "tot_outflow": [10 + i for i in range(n)],
        }
    )


def _distance_matrix(tessellation: pd.DataFrame) -> np.ndarray:
    from skmob2.models.gravity import compute_distance_matrix

    return compute_distance_matrix(tessellation, np.arange(len(tessellation)))


def _native_frame(df):
    return df.df if hasattr(df, "df") else df


def _expected_gravity(tessellation, deterrence, gravity_type, out_format, origin_exp=1.5, destination_exp=2.0):
    relevance = tessellation["relevance"].to_numpy(dtype=float)
    outflows = tessellation["tot_outflow"].to_numpy(dtype=float)
    distance = _distance_matrix(tessellation)
    scores = deterrence(distance) * relevance[None, :] ** destination_exp * relevance[:, None] ** origin_exp
    np.fill_diagonal(scores, 0.0)
    np.putmask(scores, np.isinf(scores), 0.0)
    if gravity_type == "globally constrained":
        probs = scores / scores.sum()
        matrix = probs * outflows.sum() if out_format == "flows" else probs
    else:
        probs = (scores.T / scores.sum(axis=1)).T
        matrix = (probs.T * outflows).T if out_format == "flows" else probs
    return matrix


@pytest.mark.parametrize(
    "deterrence_func_type,args,deterrence",
    [
        ("power_law", [-2.0], lambda x: powerlaw_deterrence_func(x, -2.0)),
        ("exponential", [0.2], lambda x: exponential_deterrence_func(x, 0.2)),
    ],
)
@pytest.mark.parametrize("gravity_type", ["singly constrained", "globally constrained"])
@pytest.mark.parametrize("out_format", ["flows", "probabilities"])
def test_gravity_generate_matches_formula(deterrence_func_type, args, deterrence, gravity_type, out_format):
    tess = _tessellation()
    model = Gravity(
        deterrence_func_type=deterrence_func_type,
        deterrence_func_args=args,
        origin_exp=1.5,
        destination_exp=2.0,
        gravity_type=gravity_type,
    )

    result = model.generate(tess, out_format=out_format)

    assert list(result.columns) == ["origin", "destination", "flow"]
    np.testing.assert_allclose(result.to_matrix(), _expected_gravity(tess, deterrence, gravity_type, out_format))


def test_core_gravity_kernel_matches_formula():
    pytest.importorskip("skmob2._core")
    from skmob2 import _core

    tess = _tessellation()
    expected = _expected_gravity(tess, lambda x: powerlaw_deterrence_func(x, -2.0), "singly constrained", "flows")

    flat = _core.model_gravity_matrix_numpy(
        tess["lat"].to_numpy(dtype=float),
        tess["lng"].to_numpy(dtype=float),
        tess["relevance"].to_numpy(dtype=float),
        tess["tot_outflow"].to_numpy(dtype=float),
        "power_law",
        -2.0,
        1.5,
        2.0,
        "singly constrained",
        "flows",
    )

    np.testing.assert_allclose(np.asarray(flat).reshape((len(tess), len(tess))), expected)


def test_core_gravity_od_row_kernel_matches_formula():
    pytest.importorskip("skmob2._core")
    from skmob2 import _core

    tess = _tessellation()
    expected = _expected_gravity(
        tess,
        lambda x: powerlaw_deterrence_func(x, -2.0),
        "singly constrained",
        "probabilities",
    )[1]

    actual = _core.model_gravity_od_row_numpy(
        1,
        tess["lat"].to_numpy(dtype=float),
        tess["lng"].to_numpy(dtype=float),
        tess["relevance"].to_numpy(dtype=float),
        "power_law",
        -2.0,
        1.5,
        2.0,
    )

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_gravity_flows_sample_is_seeded():
    tess = _tessellation()
    model = Gravity(gravity_type="singly constrained")

    np.random.seed(123)
    first = model.generate(tess, out_format="flows_sample")
    np.random.seed(123)
    second = model.generate(tess, out_format="flows_sample")

    pd.testing.assert_frame_equal(_native_frame(first), _native_frame(second))


@pytest.mark.parametrize("out_format", ["flows", "flows_sample", "probabilities"])
def test_radiation_generate_shapes(out_format):
    tess = _tessellation()
    np.random.seed(7)

    result = Radiation().generate(tess, out_format=out_format)

    result = _native_frame(result)
    assert list(result.columns) == ["origin", "destination", "flow"]
    assert set(result["origin"]).issubset(set(tess["tile_id"]))
    assert set(result["destination"]).issubset(set(tess["tile_id"]))
    if out_format == "probabilities":
        sums = result.groupby("origin")["flow"].sum().to_numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums))


def test_core_radiation_kernel_matches_python_probabilities():
    pytest.importorskip("skmob2._core")
    from skmob2 import _core

    tess = _tessellation()
    model = Radiation()
    model._out_format = "probabilities"
    model._tile_id_column = "tile_id"
    model.lats_lngs = tess[["lat", "lng"]].to_numpy(dtype=float)
    model.relevances = tess["relevance"].to_numpy(dtype=float)
    expected = model._from_matrix_to_flowdf(
        [row for origin in range(len(tess)) for row in model._get_flows(origin, model.relevances.sum())],
        tess,
    )

    origins, destinations, probabilities = _core.model_radiation_probabilities(
        tess["lat"].to_numpy(dtype=float),
        tess["lng"].to_numpy(dtype=float),
        tess["relevance"].to_numpy(dtype=float),
        np.ones(len(tess), dtype=float),
    )
    actual = pd.DataFrame(
        {
            "origin": [tess.loc[i, "tile_id"] for i in origins],
            "destination": [tess.loc[i, "tile_id"] for i in destinations],
            "flow": probabilities,
        }
    )

    pd.testing.assert_frame_equal(
        actual,
        _native_frame(expected),
        check_dtype=False,
        check_frame_type=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_gravity_accepts_narwhals_compatible_dataframe():
    nw = pytest.importorskip("narwhals")
    tess = nw.from_native(_tessellation())

    result = Gravity().generate(tess)

    assert len(result) > 0
    assert list(result.columns) == ["origin", "destination", "flow"]


def test_markov_diary_generator_fit_and_generate():
    start = pd.Timestamp("2020-01-01 00:00:00")
    traj = pd.DataFrame(
        {
            "uid": [1, 1, 1, 1, 2, 2, 2],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 03:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                ]
            ),
            "cluster": [0, 1, 0, 1, 0, 0, 1],
        }
    )
    mdg = MarkovDiaryGenerator()
    mdg.fit(traj, 2, lid="cluster")

    diary = mdg.generate(8, start, random_state=0)

    assert list(diary.columns) == ["datetime", "abstract_location"]
    assert len(diary) >= 1
    assert diary["datetime"].is_monotonic_increasing


@pytest.mark.parametrize("model_cls", [EPR, DensityEPR, SpatialEPR])
def test_epr_family_generates_trajectory(model_cls):
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = model_cls().generate(start, end, _tessellation(), n_agents=2, random_state=0)

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert set(result["uid"]) == {1, 2}
    assert result["datetime"].min() == start


def test_epr_random_state_is_reproducible():
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    first = EPR().generate(start, end, _tessellation(), n_agents=3, random_state=123).df
    second = EPR().generate(start, end, _tessellation(), n_agents=3, random_state=123).df

    pd.testing.assert_frame_equal(first, second)


def test_epr_starting_locations_are_used_in_order():
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 01:00:00")
    tess = _tessellation()

    result = EPR().generate(start, end, tess, n_agents=2, starting_locations=[2, 1], random_state=0).df
    first_points = result.sort_values(["uid", "datetime"]).groupby("uid", as_index=False).first()

    np.testing.assert_allclose(first_points["lat"].to_numpy(), tess.loc[[2, 1], "lat"].to_numpy())
    np.testing.assert_allclose(first_points["lng"].to_numpy(), tess.loc[[2, 1], "lng"].to_numpy())


def test_epr_polars_tessellation_returns_polars_wrapped_frame():
    pytest.importorskip("powerlaw")
    pl = pytest.importorskip("polars")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = EPR().generate(start, end, pl.from_pandas(_tessellation()), n_agents=2, random_state=0)

    assert isinstance(result.df, pl.DataFrame)
    assert result.columns == ["uid", "datetime", "lat", "lng"]


def test_geosim_generates_trajectory():
    pytest.importorskip("powerlaw")
    pytest.importorskip("igraph")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = GeoSim().generate(start, end, _tessellation(3), n_agents=2, random_state=0)

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert len(result) >= 2


def test_sts_epr_generates_trajectory():
    pytest.importorskip("igraph")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    mdg = MarkovDiaryGenerator()

    result = STS_epr().generate(
        start, end, _tessellation(3), mdg, n_agents=2, random_state=0, relevance_column="relevance"
    )

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert len(result) >= 2


def test_cluster_imports_without_scikit_learn_until_called(monkeypatch):
    cluster_mod = importlib.import_module("skmob2.preprocessing._cluster")

    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)

    with pytest.raises(ImportError, match=r"pip install skmob2\[ai\]"):
        cluster_mod._dbscan_cls()


@pytest.mark.skmob
def test_gravity_matches_skmob_when_available():
    try:
        from skmob.models import Gravity as SkmobGravity
    except Exception as exc:  # pragma: no cover - depends on optional skmob environment
        pytest.skip(f"skmob is not importable: {exc}")

    tess = _tessellation()
    ours = Gravity().generate(tess, out_format="probabilities").to_matrix()
    theirs = SkmobGravity().generate(tess, out_format="probabilities").to_matrix()

    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)
