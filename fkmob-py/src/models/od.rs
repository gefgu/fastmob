use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use fkmob_core::models::od::{gravity_matrix_impl, gravity_od_row_impl};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (latitudes, longitudes, relevances, tot_outflows, deterrence_type, deterrence_arg, origin_exp, destination_exp, gravity_type, out_format))]
pub fn model_gravity_matrix_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    tot_outflows: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    gravity_type: &str,
    out_format: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let relevances = relevances.as_slice()?;
    let tot_outflows = tot_outflows.as_slice()?;
    let matrix = gravity_matrix_impl(
        lats,
        lons,
        relevances,
        tot_outflows,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
        gravity_type,
        out_format,
    )
    .map_err(PyValueError::new_err)?;
    Ok(matrix.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn model_gravity_od_row_numpy<'py>(
    py: Python<'py>,
    origin: usize,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let row = gravity_od_row_impl(
        origin,
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        relevances.as_slice()?,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
    )
    .map_err(PyValueError::new_err)?;
    Ok(row.into_pyarray(py))
}
