mod cdr;
mod clustering;
mod compress_traj;
mod entropy;
mod filter_traj;
mod haversine;
mod home_location;
mod jump_lengths;
mod k_radius_of_gyration;
mod location_frequency;
mod max_distance_from_point;
mod maximum_distance;
mod motifs;
mod radius_of_gyration;
mod recency_rank;
mod spatial_counts;
mod square_displacement;
mod stay_locations_rs;
mod stvd_emd;
mod total_distance;
mod uncorrelated_entropy;
mod utils;
mod visitation_law;
mod waiting_times;
mod wasserstein;

use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_km, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_flat_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths::jump_lengths_flat_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_indexed_flat_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_indexed_flat_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_flat_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_flat_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_single_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_single_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_single_flat_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::jump_lengths_time_ordered_single_flat_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::time_ordered_user_indices_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::time_ordered_user_indices_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::time_ordered_single_user_indices_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        jump_lengths::time_ordered_single_user_indices_arrow,
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
        radius_of_gyration::radius_of_gyration_arrow,
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
        k_radius_of_gyration::k_radius_of_gyration_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_indexed_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_indexed_arrow,
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
        max_distance_from_point::max_distance_from_point_indexed_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_indexed_arrow,
        m
    )?)?;
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
    m.add_function(wrap_pyfunction!(filter_traj::filter_trajectory_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj::filter_trajectory_indices_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj::filter_trajectory_indices_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj::compress_trajectory_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj::compress_trajectory_representatives,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj::compress_trajectory_representatives_numpy,
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
        stay_locations_rs::detect_stay_locations_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_rs::detect_stay_locations_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein::wasserstein_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein::wasserstein_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_gmm, m)?)?;
    Ok(())
}
