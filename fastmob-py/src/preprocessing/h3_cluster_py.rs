use fastmob_core::preprocessing::h3_cluster::h3_connected_components;
use h3o::Resolution;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_u64_values, arrow_values, as_f64_array, as_u64_array, i32_results_into_arrow,
};

/// Arrow-only H3 connected-component labels. This avoids a NumPy conversion
/// for every dataframe backend before entering the Rust kernel.
#[pyfunction]
pub fn h3_cluster_labels_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    group_ends: ArrowPyArray,
    resolution: u8,
    min_samples: usize,
) -> PyResult<ArrowPyArray> {
    let resolution = Resolution::try_from(resolution)
        .map_err(|_| PyValueError::new_err("H3 resolution must be between 0 and 15"))?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ends = as_u64_array(group_ends, "group_ends")?;
    let ends: Vec<usize> = arrow_u64_values(&ends)
        .iter()
        .map(|&end| {
            usize::try_from(end)
                .map_err(|_| PyValueError::new_err("group ends exceed platform size"))
        })
        .collect::<PyResult<_>>()?;
    let labels = py
        .detach(|| {
            h3_connected_components(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                &ends,
                resolution,
                min_samples,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(i32_results_into_arrow(labels))
}
