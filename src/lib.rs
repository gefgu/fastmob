#![warn(clippy::disallowed_types)]

mod measures;
mod models;
mod preprocessing;
mod utils;

use pyo3::prelude::*;

use measures::collective::{square_displacement, visitation_law};
use measures::evaluation::{stvd_emd, wasserstein};
pub(crate) use measures::individual::time_ordering;
use measures::individual::{
    entropy, home_location, k_radius_of_gyration, location_frequency, max_distance_from_point,
    maximum_distance, motifs, radius_of_gyration, recency_rank, spatial_counts, total_distance,
    uncorrelated_entropy, waiting_times,
};
use preprocessing::{
    cdr, clustering, compress_traj_py, filter_traj, filter_traj_py, stay_locations_py,
};
pub(crate) use utils::haversine;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(
        visitation_law::visitation_distances_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        visitation_law::visitation_distances_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        visitation_law::visitation_distances_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths::jump_lengths_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths_numpy::jump_lengths_presorted_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths_arrow::jump_lengths_presorted_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths_numpy::jump_lengths_non_ordered_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths_arrow::jump_lengths_non_ordered_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        time_ordering::time_ordered_user_indices_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        time_ordering::time_ordered_user_indices_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_batch_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_numpy_with_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_arrow_with_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_user_indices_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_user_indices_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_valid_user_indices_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_valid_user_indices_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_batch_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        total_distance::total_distance_batch_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(total_distance::total_distance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(total_distance::total_distance_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        total_distance::total_distance_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        total_distance::total_distance_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(spatial_counts::number_of_visits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_counts::number_of_visits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::location_frequency_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::location_frequency_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        recency_rank::recency_rank_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        recency_rank::recency_rank_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_batch_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(home_location::home_location_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(home_location::home_location_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        home_location::home_location_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        home_location::home_location_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_flat_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_flat_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_indexed_flat_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_indexed_flat_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        square_displacement::square_displacement_km2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        square_displacement::mean_square_displacement_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        square_displacement::mean_square_displacement_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        uncorrelated_entropy::uncorrelated_entropy_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        uncorrelated_entropy::uncorrelated_entropy_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(motifs::canonical_adjacency_form, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::compute_daily_motifs, m)?)?;
    m.add_class::<filter_traj::FilterConfig>()?;
    m.add_function(wrap_pyfunction!(filter_traj_py::filter_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj_py::filter_trajectory_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj_py::filter_trajectory_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj_py::filter_trajectory_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj_py::filter_trajectory_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_representatives_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_representatives_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_representatives_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_representatives_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_approx_travel_minutes, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_visitation_stays, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_trip_indices, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::trajectory_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        entropy::trajectory_predictability_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(entropy::real_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_py::detect_stay_locations_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_py::detect_stay_locations_batch_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_py::detect_stay_locations_batch_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_py::detect_stay_locations_batch_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein::wasserstein_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein::wasserstein_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_gravity_matrix_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_gravity_od_row_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_radiation_probabilities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_truncated_power_law_samples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_distance_matrix_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_epr_simulate_agents,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::model_generation::model_epr_simulate_agents_from_od,
        m
    )?)?;
    Ok(())
}
