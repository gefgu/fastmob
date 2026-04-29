mod motifs;
mod haversine;
mod jump_lengths;
mod k_radius_of_gyration;
mod max_distance_from_point;
mod maximum_distance;
mod radius_of_gyration;
mod square_displacement;
mod total_distance;
mod waiting_times;
mod filter_traj;
mod compress_traj;
mod cdr;
mod entropy;
mod stay_locations_rs;
mod stvd_emd;
mod clustering;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_km, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_km, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_user_indices_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_user_indices_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_indexed_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration::radius_of_gyration_indexed_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(maximum_distance::maximum_distance_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(total_distance::total_distance_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(k_radius_of_gyration::k_radius_of_gyration_km, m)?)?;
    m.add_function(wrap_pyfunction!(max_distance_from_point::max_distance_from_point_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(square_displacement::square_displacement_km2, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::canonical_adjacency_form, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::compute_daily_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(filter_traj::filter_trajectory_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compress_traj::compress_trajectory_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_approx_travel_minutes, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_visitation_stays, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_trip_indices, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::trajectory_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::trajectory_predictability_batch, m)?)?;
    m.add_function(wrap_pyfunction!(stay_locations_rs::detect_stay_locations_batch, m)?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_gmm, m)?)?;
    Ok(())
}
