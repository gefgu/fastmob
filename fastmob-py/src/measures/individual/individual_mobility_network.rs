use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::measures::individual::individual_mobility_network::{
    individual_mobility_network_indexed_impl, individual_mobility_network_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow,
};

type MobilityNetwork<'py> = (
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    Bound<'py, PyArray1<u64>>,
);

#[pyfunction]
pub fn individual_mobility_network_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    self_loops: bool,
) -> PyResult<MobilityNetwork<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
        individual_mobility_network_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            &indices.as_slice()?,
            &ends.as_slice()?,
            self_loops,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(lat_origins),
        f64_results_into_arrow(lng_origins),
        f64_results_into_arrow(lat_dests),
        f64_results_into_arrow(lng_dests),
        u64_results_into_arrow(n_trips),
        user_indices
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}

#[pyfunction]
pub fn individual_mobility_network_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    self_loops: bool,
) -> PyResult<MobilityNetwork<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (lat_origins, lng_origins, lat_dests, lng_dests, n_trips, user_indices) =
        individual_mobility_network_presorted_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            &ends.as_slice()?,
            self_loops,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(lat_origins),
        f64_results_into_arrow(lng_origins),
        f64_results_into_arrow(lat_dests),
        f64_results_into_arrow(lng_dests),
        u64_results_into_arrow(n_trips),
        user_indices
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}
