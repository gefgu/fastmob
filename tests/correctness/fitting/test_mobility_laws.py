"""Correctness tests for the H3 universal visitation law."""

from __future__ import annotations

import numpy as np
import pytest


def _staypoints_fixture():
    import pandas as pd

    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "timestamp": pd.to_datetime(
                ["2020-01-01 23:00", "2020-01-02 09:00", "2020-01-03 09:00", "2020-01-01 23:00", "2020-01-02 09:00"]
            ),
            "lat": [0.0, 0.0, 0.0, 1.0, 1.0],
            "lng": [0.0, 1.0, 1.0, 0.0, 1.0],
        }
    )


def test_fit_visitation_law_returns_data_spectrum_and_parameters():
    from fastmob.measures.fitting import VisitationLawFit, fit_visitation_law

    result = fit_visitation_law(_staypoints_fixture(), h3_resolution=5)

    assert isinstance(result, VisitationLawFit)
    assert list(result.data.columns) == ["user_id", "h3_cell", "r_km", "f", "rf", "n_staypoints"]
    assert list(result.spectrum.columns) == ["rf", "rho"]
    assert len(result.data) == 4
    assert len(result.spectrum) >= 2
    assert np.isfinite(result.eta)
    assert np.isfinite(result.mu)
    assert np.isfinite(result.r2)


def test_fit_visitation_law_accepts_staypoints():
    from fastmob.core import Staypoints
    from fastmob.measures.fitting import fit_visitation_law

    frame = _staypoints_fixture().rename(columns={"timestamp": "started_at"})
    frame["finished_at"] = frame["started_at"] + __import__("pandas").Timedelta(minutes=10)
    expected = fit_visitation_law(frame.rename(columns={"started_at": "timestamp"}), h3_resolution=5)
    result = fit_visitation_law(Staypoints(frame), h3_resolution=5)

    expected_data = expected.data.sort_values(["user_id", "h3_cell"]).reset_index(drop=True)
    result_data = result.data.sort_values(["user_id", "h3_cell"]).reset_index(drop=True)
    assert result_data[["user_id", "h3_cell", "f", "n_staypoints"]].equals(
        expected_data[["user_id", "h3_cell", "f", "n_staypoints"]]
    )
    np.testing.assert_allclose(result_data["r_km"], expected_data["r_km"])


def test_fit_visitation_law_honours_timezone_and_invalid_coordinates():
    from fastmob.measures.fitting import fit_visitation_law

    staypoints = _staypoints_fixture()
    local = staypoints.copy()
    local["timestamp"] = local["timestamp"].dt.tz_localize("Europe/Lisbon")
    expected = fit_visitation_law(staypoints, h3_resolution=5)
    result = fit_visitation_law(local, h3_resolution=5)
    np.testing.assert_allclose(expected.data.sort_values("h3_cell")["r_km"], result.data.sort_values("h3_cell")["r_km"])

    staypoints.loc[len(staypoints)] = ["invalid", staypoints["timestamp"].iloc[0], np.nan, 0.0]
    filtered = fit_visitation_law(staypoints, h3_resolution=5)
    assert set(filtered.data["user_id"]) == {"u1", "u2"}


def test_fit_visitation_law_validation_and_backend_parity():
    pl = pytest.importorskip("polars")
    from fastmob.measures.fitting import fit_visitation_law

    staypoints = _staypoints_fixture()
    with pytest.raises(ValueError, match="h3_resolution"):
        fit_visitation_law(staypoints, h3_resolution=16)

    pandas_result = fit_visitation_law(staypoints, h3_resolution=5)
    polars_result = fit_visitation_law(pl.from_pandas(staypoints), h3_resolution=5)
    pandas_data = pandas_result.data.sort_values(["user_id", "h3_cell"]).reset_index(drop=True)
    polars_data = polars_result.data.to_pandas().sort_values(["user_id", "h3_cell"]).reset_index(drop=True)
    assert pandas_data[["user_id", "h3_cell", "f", "n_staypoints"]].equals(
        polars_data[["user_id", "h3_cell", "f", "n_staypoints"]]
    )
    np.testing.assert_allclose(pandas_data["r_km"], polars_data["r_km"])


def test_visitation_distance_core_helper_accepts_numpy_and_arrow():
    pl = pytest.importorskip("polars")
    from fastmob._core import visitation_distances, visitation_distances_km

    home_lats = np.array([0.0, 10.0], dtype=np.float64)
    home_lngs = np.array([0.0, 0.0], dtype=np.float64)
    loc_lats = np.array([0.0, 10.0], dtype=np.float64)
    loc_lngs = np.array([1.0, 2.0], dtype=np.float64)

    expected = visitation_distances_km(home_lats.tolist(), home_lngs.tolist(), loc_lats.tolist(), loc_lngs.tolist())
    result_numpy = visitation_distances(home_lats, home_lngs, loc_lats, loc_lngs)
    result_arrow = visitation_distances(
        pl.Series(home_lats).to_arrow(),
        pl.Series(home_lngs).to_arrow(),
        pl.Series(loc_lats).to_arrow(),
        pl.Series(loc_lngs).to_arrow(),
    )

    np.testing.assert_allclose(result_numpy, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result_arrow, expected, rtol=0.0, atol=1e-12)


def test_visitation_distance_core_helper_validation_errors():
    from fastmob._core import visitation_distances

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        visitation_distances(arr, arr, arr[:1], arr)


def test_visitation_law_arrow_binning_factorizes_string_ids_and_filters_invalid_rows():
    import pyarrow as pa

    from fastmob._core import bin_visitation_law_arrow

    rf, rho = bin_visitation_law_arrow(
        pa.array(["u1", "u1", "u2", "u2", "ignored"]),
        pa.array(["a", "a", "a", "b", "a"]),
        pa.array([0.2, 0.2, 0.2, 1.2, np.nan], type=pa.float64()),
        pa.array([2.0, 2.0, 2.0, 3.0, 1.0], type=pa.float64()),
        2,
        1.0,
    )

    np.testing.assert_allclose(np.asarray(rf), [4.5**0.25, 4.5**0.75])
    np.testing.assert_allclose(np.asarray(rho), [2.0 / np.pi, 1.0 / (3.0 * np.pi)])


def test_visitation_law_arrow_binning_validates_arrow_column_lengths():
    import pyarrow as pa

    from fastmob._core import bin_visitation_law_arrow

    with pytest.raises(ValueError, match="same length"):
        bin_visitation_law_arrow(
            pa.array(["u1", "u2"]),
            pa.array(["a"]),
            pa.array([1.0, 2.0], type=pa.float64()),
            pa.array([1.0, 2.0], type=pa.float64()),
            2,
            1.0,
        )
