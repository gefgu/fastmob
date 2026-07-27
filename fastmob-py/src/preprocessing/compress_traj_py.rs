use fastmob_core::preprocessing::compress_traj::{
    compress_trajectory_representatives_impl, compress_trajectory_representatives_indexed_impl,
    compress_user_slice,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow, validate_indexed_ends,
};

type CompressRepresentatives<'py> = (Py<PyAny>, Py<PyAny>, Py<PyAny>);

fn arrow_usize_output(py: Python<'_>, values: Vec<usize>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        u64_results_into_arrow(values.into_iter().map(|i| i as u64).collect()),
    )?
    .into_any())
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn compress_trajectory_batch(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let mut all_groups: Vec<(usize, usize)> = Vec::new();

    for &(start, end) in &ranges {
        let user_groups = compress_user_slice(
            &latitudes[start..end],
            &longitudes[start..end],
            spatial_radius_km,
            start,
        );
        all_groups.extend(user_groups);
    }

    Ok(all_groups)
}

#[pyfunction]
pub fn compress_trajectory_representatives(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentatives<'_>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            &ranges,
            spatial_radius_km,
        )
    });
    Ok((
        arrow_usize_output(py, representative_indices)?,
        arrow_f64_output(py, median_latitudes)?,
        arrow_f64_output(py, median_longitudes)?,
    ))
}

#[pyfunction]
pub fn compress_trajectory_representatives_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentatives<'py>> {
    let sorted_indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;

    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            sorted_indices,
            ends,
            valid_rows.as_deref(),
            spatial_radius_km,
        )
    });
    Ok((
        arrow_usize_output(py, representative_indices)?,
        arrow_f64_output(py, median_latitudes)?,
        arrow_f64_output(py, median_longitudes)?,
    ))
}
