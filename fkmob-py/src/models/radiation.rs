use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use fkmob_core::models::radiation::model_radiation_probabilities_impl;

#[pyfunction]
pub fn model_radiation_probabilities(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    relevances: PyReadonlyArray1<f64>,
    tot_outflows: PyReadonlyArray1<f64>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    model_radiation_probabilities_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        relevances.as_slice()?,
        tot_outflows.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}
