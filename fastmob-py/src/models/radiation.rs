use fastmob_core::models::radiation::{
    model_radiation_probabilities_impl, model_radiation_sample_flows_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array, f64_results_into_arrow, u64_results_into_arrow};

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

#[pyfunction]
pub fn model_radiation_flows_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    tot_outflows: ArrowPyArray,
    out_format: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let relevances = as_f64_array(relevances, "relevances")?;
    let tot_outflows = as_f64_array(tot_outflows, "tot_outflows")?;
    let (origins, destinations, mut values) = py
        .detach(|| {
            model_radiation_probabilities_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&relevances),
                arrow_values(&tot_outflows),
            )
        })
        .map_err(PyValueError::new_err)?;
    if out_format == "flows" {
        for (index, value) in values.iter_mut().enumerate() {
            *value = (tot_outflows.value(origins[index]) * *value).round();
        }
    }
    let keep: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, value)| (*value > 0.0).then_some(i))
        .collect();
    Ok((
        Py::new(
            py,
            u64_results_into_arrow(keep.iter().map(|&i| origins[i] as u64).collect()),
        )?
        .into_any(),
        Py::new(
            py,
            u64_results_into_arrow(keep.iter().map(|&i| destinations[i] as u64).collect()),
        )?
        .into_any(),
        Py::new(
            py,
            f64_results_into_arrow(keep.iter().map(|&i| values[i]).collect()),
        )?
        .into_any(),
    ))
}

#[pyfunction]
pub fn model_radiation_sample_flows_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    tot_outflows: ArrowPyArray,
    seed: u64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let relevances = as_f64_array(relevances, "relevances")?;
    let outflows = as_f64_array(tot_outflows, "tot_outflows")?;
    let (origins, destinations, values) = py
        .detach(|| {
            model_radiation_sample_flows_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&relevances),
                arrow_values(&outflows),
                seed,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        Py::new(
            py,
            u64_results_into_arrow(origins.into_iter().map(|value| value as u64).collect()),
        )?
        .into_any(),
        Py::new(
            py,
            u64_results_into_arrow(destinations.into_iter().map(|value| value as u64).collect()),
        )?
        .into_any(),
        Py::new(py, f64_results_into_arrow(values))?.into_any(),
    ))
}
