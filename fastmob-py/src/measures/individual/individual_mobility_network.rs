use fastmob_core::measures::individual::individual_mobility_network::{
    individual_mobility_network_indexed_impl, individual_mobility_network_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow,
};

type MobilityNetwork<'py> = (
    PyArray,
    PyArray,
    PyArray,
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
);

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
pub fn individual_mobility_network_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    self_loops: bool,
) -> PyResult<MobilityNetwork<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
            individual_mobility_network_indexed_impl(
                latitudes.as_slice()?,
                longitudes.as_slice()?,
                indices.as_slice()?,
                ends.as_slice()?,
                self_loops,
                None,
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(lat_origins),
            f64_results_into_arrow(lng_origins),
            f64_results_into_arrow(lat_dests),
            f64_results_into_arrow(lng_dests),
            u64_results_into_arrow(n_trips),
            user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<PyArray>()?, "latitudes")?;
        let longitudes = as_nullable_f64_array(longitudes.extract::<PyArray>()?, "longitudes")?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
            individual_mobility_network_indexed_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                indices.as_slice()?,
                ends.as_slice()?,
                self_loops,
                valid_rows.as_deref(),
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(lat_origins),
            f64_results_into_arrow(lng_origins),
            f64_results_into_arrow(lat_dests),
            f64_results_into_arrow(lng_dests),
            u64_results_into_arrow(n_trips),
            user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn individual_mobility_network_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    self_loops: bool,
) -> PyResult<MobilityNetwork<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
            individual_mobility_network_presorted_impl(
                latitudes.as_slice()?,
                longitudes.as_slice()?,
                ends.as_slice()?,
                self_loops,
                None,
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(lat_origins),
            f64_results_into_arrow(lng_origins),
            f64_results_into_arrow(lat_dests),
            f64_results_into_arrow(lng_dests),
            u64_results_into_arrow(n_trips),
            user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<PyArray>()?, "latitudes")?;
        let longitudes = as_nullable_f64_array(longitudes.extract::<PyArray>()?, "longitudes")?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
            individual_mobility_network_presorted_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                ends.as_slice()?,
                self_loops,
                valid_rows.as_deref(),
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(lat_origins),
            f64_results_into_arrow(lng_origins),
            f64_results_into_arrow(lat_dests),
            f64_results_into_arrow(lng_dests),
            u64_results_into_arrow(n_trips),
            user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}
