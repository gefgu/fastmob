#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod adapters;
mod hierarchy_py;
mod integration;
mod measures;
mod models;
mod network;
mod preprocessing;
mod privacy;
mod trajectory;
mod utils;

use pyo3::prelude::*;

use integration::events_py;
use measures::collective::{co_presence_network, square_displacement, visitation_law};
#[cfg(feature = "stvd-emd")]
use measures::evaluation::stvd_emd;
use measures::evaluation::{trajectory_cpc, wasserstein};
use measures::fitting::truncated_powerlaw as fitting_truncated_powerlaw;
use measures::individual::{
    activity, diversity, entropy, factorization, home_location, indexed_user_indices,
    individual_mobility_network, k_radius_of_gyration, location_frequency, max_distance_from_point,
    maximum_distance, motifs, radius_of_gyration, recency_rank, spatial_counts, time_ordering,
    total_distance, uncorrelated_entropy, waiting_times,
};
use network::road_graph_py;
use preprocessing::{
    cdr, clustering, compress_traj_py, filter_traj_py, h3_py, outliers_traj_py, segment_traj_py,
    simplify_traj_py, stay_locations_py,
};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(activity::activity_counts, m)?)?;
    m.add_function(wrap_pyfunction!(factorization::factorize_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(activity::activity_transition_counts, m)?)?;
    m.add_function(wrap_pyfunction!(activity::daily_activity_percentages, m)?)?;
    m.add_function(wrap_pyfunction!(utils::haversine_py::haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(utils::haversine_py::haversine_m_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        visitation_law::visitation_distances_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(visitation_law::visitation_distances, m)?)?;
    m.add_function(wrap_pyfunction!(
        fitting_truncated_powerlaw::fit_truncated_powerlaw_grid,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths::jump_lengths_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths::jump_lengths_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        measures::individual::jump_lengths::jump_lengths_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        time_ordering::time_ordered_user_indices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        indexed_user_indices::indexed_user_indices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        indexed_user_indices::single_user_indices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        radius_of_gyration::radius_of_gyration_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        maximum_distance::maximum_distance_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        time_ordering::presorted_user_starts_ends,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        total_distance::total_distance_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(total_distance::total_distance_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_km,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        k_radius_of_gyration::k_radius_of_gyration_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_visits_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        spatial_counts::number_of_locations_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::location_frequency_values_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::location_frequency_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::frequency_rank_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        location_frequency::frequency_rank_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        recency_rank::recency_rank_values_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(recency_rank::recency_rank_presorted, m)?)?;
    m.add_function(wrap_pyfunction!(
        individual_mobility_network::individual_mobility_network_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        individual_mobility_network::individual_mobility_network_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_distance_from_point::max_distance_from_point_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(home_location::home_location_presorted, m)?)?;
    m.add_function(wrap_pyfunction!(home_location::home_location_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_presorted, m)?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_presorted_flat,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(waiting_times::waiting_times_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(
        waiting_times::waiting_times_indexed_flat,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        square_displacement::square_displacement_km2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        square_displacement::mean_square_displacement_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        uncorrelated_entropy::uncorrelated_entropy_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        co_presence_network::build_co_presence_edges_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(co_presence_network::graph_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        co_presence_network::random_baseline_overlap_threshold_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(motifs::canonical_adjacency_form, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::encode_motif_purposes, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::daily_motifs_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::daily_motifs_presorted, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::daily_motifs_indexed_joined, m)?)?;
    m.add_function(wrap_pyfunction!(motifs::daily_motifs_presorted_joined, m)?)?;
    m.add_function(wrap_pyfunction!(
        hierarchy_py::tripleg_lengths_attributed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(hierarchy_py::trips_from_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchy_py::tours_from_trips, m)?)?;
    m.add_class::<road_graph_py::RoadNetworkHandle>()?;
    m.add_function(wrap_pyfunction!(
        road_graph_py::subsample_waypoints_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        events_py::nearest_event_within_window_numpy,
        m
    )?)?;
    m.add_class::<preprocessing::filter_traj_py::PyFilterConfig>()?;
    m.add_function(wrap_pyfunction!(filter_traj_py::filter_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(
        filter_traj_py::filter_trajectory_indexed,
        m
    )?)?;
    m.add_class::<preprocessing::outliers_traj_py::PyOutlierConfig>()?;
    m.add_function(wrap_pyfunction!(
        outliers_traj_py::outlier_trajectory_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compress_traj_py::compress_trajectory_representatives_indexed,
        m
    )?)?;
    m.add_class::<preprocessing::simplify_traj_py::PySimplifyConfig>()?;
    m.add_function(wrap_pyfunction!(
        simplify_traj_py::simplify_trajectory_indexed,
        m
    )?)?;
    m.add_class::<preprocessing::segment_traj_py::PySegmentConfig>()?;
    m.add_function(wrap_pyfunction!(
        segment_traj_py::segment_trajectory_indexed,
        m
    )?)?;
    m.add_class::<trajectory::interpolate_py::PyInterpolationConfig>()?;
    m.add_function(wrap_pyfunction!(
        trajectory::interpolate_py::interpolate_trajectory_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::interpolate_py::interpolate_trajectory_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::interpolate_at_py::interpolate_at_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::interpolate_at_py::interpolate_at_presorted,
        m
    )?)?;
    m.add_class::<trajectory::distance_py::PyDistanceConfig>()?;
    m.add_function(wrap_pyfunction!(
        trajectory::distance_py::trajectory_distance,
        m
    )?)?;
    m.add_class::<trajectory::smooth_py::PySmoothConfig>()?;
    m.add_function(wrap_pyfunction!(
        trajectory::smooth_py::smooth_trajectory_indexed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::smooth_py::smooth_trajectory_presorted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::shape_cluster_py::trajectory_shape_signatures,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        trajectory::shape_cluster_py::cluster_trajectory_shape_signatures,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_approx_travel_minutes, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_visitation_stays, m)?)?;
    m.add_function(wrap_pyfunction!(cdr::cdr_trip_indices, m)?)?;
    m.add_function(wrap_pyfunction!(h3_py::latlng_to_h3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(h3_py::latlng_to_h3_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(h3_py::h3_to_latlng_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        preprocessing::h3_cluster_py::h3_cluster_labels_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(entropy::trajectory_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        entropy::trajectory_predictability_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(entropy::real_entropy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::real_entropy_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(diversity::diversity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        stay_locations_py::detect_stay_locations_batch_indexed,
        m
    )?)?;
    #[cfg(feature = "stvd-emd")]
    {
        m.add_function(wrap_pyfunction!(stvd_emd::stvd_emd, m)?)?;
    }
    m.add_function(wrap_pyfunction!(
        trajectory_cpc::trajectory_common_part_of_commuters,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(wasserstein::wasserstein, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(clustering::cluster_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(models::od::model_gravity_matrix_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(models::od::model_gravity_od_row_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::radiation::model_radiation_probabilities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::epr::model_truncated_power_law_samples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::distance::model_distance_matrix_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(models::epr::model_epr_simulate_agents, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::social_graph::model_social_graph_edges_to_csr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::social_graph::model_social_graph_random_geometric,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_update_chain,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_normalize,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_build_cdf,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_fit_from_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_batch_generate,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::markov_diary::markov_diary_generate,
        m
    )?)?;
    m.add_class::<models::next_location_py::PyNextLocationModels>()?;
    m.add_function(wrap_pyfunction!(
        models::geosim::model_geosim_simulate_agents,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::sts_epr::model_sts_epr_simulate_agents,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::ditras::model_ditras_simulate_agents,
        m
    )?)?;
    m.add_class::<privacy::PyPrivacyRiskResult>()?;
    m.add_function(wrap_pyfunction!(privacy::privacy_assess_risk, m)?)?;
    Ok(())
}
