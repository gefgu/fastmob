from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from fastmob.core import Locations
from fastmob.models import (
    EPR,
    DensityEPR,
    Ditras,
    GeoSim,
    Gravity,
    MarkovDiaryGenerator,
    Radiation,
    SpatialEPR,
    STS_epr,
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


def _locations(tessellation: pd.DataFrame) -> Locations:
    return Locations.from_tessellation(tessellation, tile_id_col="tile_id", lat_col="lat", lng_col="lng")


def _powerlaw_deterrence(x, exponent):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, exponent)


def _exponential_deterrence(x, rate):
    return np.exp(-x * rate)


def _distance_matrix(tessellation: pd.DataFrame) -> np.ndarray:
    from fastmob import _core

    lats = tessellation["lat"].to_numpy(dtype=float)
    lngs = tessellation["lng"].to_numpy(dtype=float)
    n = len(tessellation)
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            distance = float(_core.haversine_km(lats[i], lngs[i], lats[j], lngs[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def _native_frame(df):
    return df.df if hasattr(df, "df") else df


def _always_away_diary_generator() -> MarkovDiaryGenerator:
    from fastmob import _core

    n_states = 48
    probs = np.zeros(n_states * n_states, dtype=float)
    for h in range(24):
        next_h = (h + 1) % 24
        probs[(h * 2 + 1) * n_states + (next_h * 2)] = 1.0
        probs[(h * 2) * n_states + (next_h * 2)] = 1.0

    mdg = MarkovDiaryGenerator()
    mdg._cdf_matrix_flat = _core.markov_diary_build_cdf(probs)
    return mdg


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
        ("power_law", [-2.0], lambda x: _powerlaw_deterrence(x, -2.0)),
        ("exponential", [0.2], lambda x: _exponential_deterrence(x, 0.2)),
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

    result = model.generate(_locations(tess), out_format=out_format)

    assert list(result.columns) == ["origin", "destination", "flow"]
    # The exponential branch's row-sum normalization is more sensitive to
    # floating-point summation order than power_law's, so the Rust kernel and
    # this independently-computed NumPy expectation diverge by ~1e-7 relative
    # even though both implement the same formula; rtol=1e-7 (the default) is
    # tight enough to catch a real formula mismatch but not this reordering
    # noise. See "skmob comparison correctness tests" in CLAUDE.md for the
    # same kind of narrow, documented tolerance relaxation.
    np.testing.assert_allclose(
        result.to_matrix(), _expected_gravity(tess, deterrence, gravity_type, out_format), rtol=1e-6
    )


def test_core_gravity_kernel_matches_formula():
    pytest.importorskip("fastmob._core")
    from fastmob import _core

    tess = _tessellation()
    expected = _expected_gravity(tess, lambda x: _powerlaw_deterrence(x, -2.0), "singly constrained", "flows")

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
    pytest.importorskip("fastmob._core")
    from fastmob import _core

    tess = _tessellation()
    expected = _expected_gravity(
        tess,
        lambda x: _powerlaw_deterrence(x, -2.0),
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
    locations = _locations(_tessellation())
    model = Gravity(gravity_type="singly constrained")

    first = model.generate(locations, out_format="flows_sample", random_state=123)
    second = model.generate(locations, out_format="flows_sample", random_state=123)

    pd.testing.assert_frame_equal(_native_frame(first), _native_frame(second))


@pytest.mark.parametrize("out_format", ["flows", "flows_sample", "probabilities"])
def test_radiation_generate_shapes(out_format):
    tess = _tessellation()

    result = Radiation().generate(_locations(tess), out_format=out_format, random_state=7)

    result = _native_frame(result)
    assert list(result.columns) == ["origin", "destination", "flow"]
    assert set(result["origin"]).issubset(set(tess["tile_id"]))
    assert set(result["destination"]).issubset(set(tess["tile_id"]))
    if out_format == "probabilities":
        sums = result.groupby("origin")["flow"].sum().to_numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums))


def test_core_radiation_kernel_matches_arrow_generate_path():
    """The raw NumPy `_core` kernel and the Arrow-native `Radiation.generate()`
    path wrap the same Rust math; there is no separate Python reference
    implementation to compare against any more, so this checks the two entry
    points agree with each other instead.
    """
    pytest.importorskip("fastmob._core")
    from fastmob import _core

    tess = _tessellation()
    expected = _native_frame(Radiation().generate(_locations(tess), out_format="probabilities"))

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
        actual.sort_values(["origin", "destination"]).reset_index(drop=True),
        expected.sort_values(["origin", "destination"]).reset_index(drop=True),
        check_dtype=False,
        check_frame_type=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_gravity_accepts_narwhals_compatible_tessellation():
    """`Gravity.generate()` now requires a `Locations` catalogue rather than a
    raw dataframe; narwhals-compatibility now lives at the
    `Locations.from_tessellation()` construction boundary instead.
    """
    nw = pytest.importorskip("narwhals")
    tess = nw.from_native(_tessellation())
    locations = Locations.from_tessellation(tess, tile_id_col="tile_id", lat_col="lat", lng_col="lng")

    result = Gravity().generate(locations)

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

    assert diary.column_names == ["datetime", "abstract_location"]
    assert len(diary) >= 1
    timestamps = diary["datetime"].to_pylist()
    assert timestamps == sorted(timestamps)


@pytest.mark.parametrize("model_cls", [EPR, DensityEPR, SpatialEPR])
def test_epr_family_generates_trajectory(model_cls):
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = model_cls().generate(start, end, _locations(_tessellation()), n_agents=2, random_state=0)

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert set(result["uid"]) == {1, 2}
    assert result["datetime"].min() == start


def test_epr_random_state_is_reproducible():
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    locations = _locations(_tessellation())

    first = EPR().generate(start, end, locations, n_agents=3, random_state=123).df
    second = EPR().generate(start, end, locations, n_agents=3, random_state=123).df

    pd.testing.assert_frame_equal(first, second)


def test_epr_starting_locations_are_used_in_order():
    pytest.importorskip("powerlaw")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 01:00:00")
    tess = _tessellation()

    result = EPR().generate(start, end, _locations(tess), n_agents=2, starting_locations=[2, 1], random_state=0).df
    first_points = result.sort_values(["uid", "datetime"]).groupby("uid", as_index=False).first()

    np.testing.assert_allclose(first_points["lat"].to_numpy(), tess.loc[[2, 1], "lat"].to_numpy())
    np.testing.assert_allclose(first_points["lng"].to_numpy(), tess.loc[[2, 1], "lng"].to_numpy())


def test_epr_polars_tessellation_returns_polars_wrapped_frame():
    pytest.importorskip("powerlaw")
    pl = pytest.importorskip("polars")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = EPR().generate(start, end, _locations(pl.from_pandas(_tessellation())), n_agents=2, random_state=0)

    assert isinstance(result.df, pl.DataFrame)
    assert result.columns == ["uid", "datetime", "lat", "lng"]


def test_geosim_generates_trajectory():
    pytest.importorskip("powerlaw")
    pytest.importorskip("igraph")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")

    result = GeoSim().generate(start, end, _locations(_tessellation(3)), n_agents=2, random_state=0)

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert len(result) >= 2


def test_sts_epr_generates_trajectory():
    pytest.importorskip("igraph")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    mdg = MarkovDiaryGenerator()

    result = STS_epr().generate(
        start, end, _locations(_tessellation(3)), mdg, n_agents=2, random_state=0, relevance_column="relevance"
    )

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert len(result) >= 2


def test_sts_epr_does_not_build_default_distance_matrix(monkeypatch):
    """When `distance_matrix=None`, `STS_epr.generate()` must pass an empty
    flat-distances array into the Rust kernel rather than materializing an
    O(n^2) matrix in Python. The Arrow-native refactor renamed the kernel
    bindings this used to patch (`model_social_graph_random_geometric` ->
    `..._arrow`, `markov_diary_batch_generate` -> `..._arrow`,
    `model_sts_epr_simulate_agents` -> `..._arrow`) and dropped the
    `model_distance_matrix_numpy` call entirely, so there is nothing left in
    that name to intercept.
    """
    import pyarrow as pa

    sts_module = importlib.import_module("fastmob.models.sts_epr")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 01:00:00")
    mdg = MarkovDiaryGenerator()
    mdg._cdf_matrix_flat = np.array([1.0], dtype=float)
    captured = {}

    def fake_simulate_agents(*args):
        captured["distances_len"] = len(args[3])
        return (
            pa.array([1], type=pa.int64()),
            np.array([45.0], dtype=np.float64),
            np.array([7.0], dtype=np.float64),
            np.array([int(start.timestamp())], dtype=np.int64),
        )

    monkeypatch.setattr(
        sts_module._core,
        "model_social_graph_random_geometric_arrow",
        lambda n_agents, radius, seed: (
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        sts_module._core,
        "markov_diary_batch_generate_arrow",
        lambda *args: (
            np.array([int(start.timestamp())], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(sts_module._core, "model_sts_epr_simulate_agents_arrow", fake_simulate_agents)

    result = STS_epr().generate(
        start,
        end,
        _locations(_tessellation(3)),
        mdg,
        n_agents=1,
        random_state=0,
        relevance_column="relevance",
    )

    assert captured["distances_len"] == 0
    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]


def test_ditras_generates_trajectory():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    mdg = MarkovDiaryGenerator()

    result = Ditras(mdg).generate(start, end, _locations(_tessellation()), n_agents=2, random_state=0)

    assert list(result.columns) == ["uid", "datetime", "lat", "lng"]
    assert set(result["uid"]) == {1, 2}
    assert result["datetime"].min() == start
    assert len(result) >= 2


def test_ditras_random_state_is_reproducible():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    mdg = MarkovDiaryGenerator()
    locations = _locations(_tessellation())

    first = Ditras(mdg).generate(start, end, locations, n_agents=3, random_state=42).df
    second = Ditras(mdg).generate(start, end, locations, n_agents=3, random_state=42).df

    pd.testing.assert_frame_equal(first, second)


def test_ditras_starting_locations_are_used():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 01:00:00")
    tess = _tessellation()
    mdg = MarkovDiaryGenerator()

    result = (
        Ditras(mdg).generate(start, end, _locations(tess), n_agents=2, starting_locations=[2, 1], random_state=0).df
    )
    first_points = result.sort_values(["uid", "datetime"]).groupby("uid", as_index=False).first()

    np.testing.assert_allclose(first_points["lat"].to_numpy(), tess.loc[[2, 1], "lat"].to_numpy())
    np.testing.assert_allclose(first_points["lng"].to_numpy(), tess.loc[[2, 1], "lng"].to_numpy())


def test_ditras_polars_tessellation_returns_polars_wrapped_frame():
    pl = pytest.importorskip("polars")
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 06:00:00")
    mdg = MarkovDiaryGenerator()

    result = Ditras(mdg).generate(start, end, _locations(pl.from_pandas(_tessellation())), n_agents=2, random_state=0)

    assert isinstance(result.df, pl.DataFrame)
    assert result.columns == ["uid", "datetime", "lat", "lng"]


def test_ditras_unfitted_diary_stays_home():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-03 00:00:00")
    tess = _tessellation()
    mdg = MarkovDiaryGenerator()  # NOT fitted → home-only CDF fallback in Rust

    result = Ditras(mdg).generate(start, end, _locations(tess), n_agents=1, starting_locations=[0], random_state=7).df

    np.testing.assert_allclose(result["lat"].to_numpy(), tess.loc[0, "lat"])
    np.testing.assert_allclose(result["lng"].to_numpy(), tess.loc[0, "lng"])


def test_ditras_exploration_uses_gravity_distance():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 02:00:00")
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1, 2],
            "lat": [45.0, 45.01, 47.0],
            "lng": [7.0, 7.01, 9.0],
            "relevance": [1.0, 1.0, 1.0],
        }
    )
    n_agents = 1200

    result = (
        Ditras(_always_away_diary_generator())
        .generate(
            start,
            end,
            _locations(tess),
            n_agents=n_agents,
            starting_locations=[0] * n_agents,
            random_state=11,
        )
        .df
    )
    away = result[result["datetime"] > start]

    near_visits = ((away["lat"] == tess.loc[1, "lat"]) & (away["lng"] == tess.loc[1, "lng"])).sum()
    far_visits = ((away["lat"] == tess.loc[2, "lat"]) & (away["lng"] == tess.loc[2, "lng"])).sum()

    assert near_visits > far_visits * 20


def test_ditras_custom_gravity_changes_exploration_distribution():
    start = pd.Timestamp("2020-01-01 00:00:00")
    end = pd.Timestamp("2020-01-01 02:00:00")
    tess = pd.DataFrame(
        {
            "tile_id": [0, 1, 2],
            "lat": [45.0, 45.01, 45.02],
            "lng": [7.0, 7.01, 7.02],
            "relevance": [1.0, 1.0, 10.0],
        }
    )
    n_agents = 1200
    mdg = _always_away_diary_generator()
    locations = _locations(tess)

    default = (
        Ditras(mdg)
        .generate(
            start,
            end,
            locations,
            n_agents=n_agents,
            starting_locations=[0] * n_agents,
            random_state=7,
        )
        .df
    )
    custom = (
        Ditras(mdg)
        .generate(
            start,
            end,
            locations,
            gravity_singly=Gravity(destination_exp=4.0, gravity_type="singly constrained"),
            n_agents=n_agents,
            starting_locations=[0] * n_agents,
            random_state=7,
        )
        .df
    )

    default_away = default[default["datetime"] > start]
    custom_away = custom[custom["datetime"] > start]
    default_far = ((default_away["lat"] == tess.loc[2, "lat"]) & (default_away["lng"] == tess.loc[2, "lng"])).sum()
    custom_far = ((custom_away["lat"] == tess.loc[2, "lat"]) & (custom_away["lng"] == tess.loc[2, "lng"])).sum()

    assert custom_far > default_far + 250


def test_cluster_imports_without_scikit_learn(monkeypatch):
    cluster_mod = importlib.import_module("fastmob.preprocessing._cluster")

    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)

    assert callable(cluster_mod.cluster)


@pytest.mark.skmob
def test_gravity_matches_skmob_when_available():
    try:
        from skmob.models import Gravity as SkmobGravity
    except Exception as exc:  # pragma: no cover - depends on optional skmob environment  # noqa: BLE001
        pytest.skip(f"skmob is not importable: {exc}")

    tess = _tessellation()
    ours = Gravity().generate(_locations(tess), out_format="probabilities").to_matrix()
    theirs = SkmobGravity().generate(tess, out_format="probabilities").to_matrix()

    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)
