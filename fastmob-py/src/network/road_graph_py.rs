use fastmob_core::network::road_graph::{
    batch_nearest_coordinates, batch_road_distances, batch_road_routes, batch_route_edge_flows,
    subsample_waypoints, RoadGraph,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_i64_values, arrow_values, as_f64_array, as_i64_array, bool_results_into_arrow,
    f64_results_into_arrow, i64_results_into_arrow, u64_results_into_arrow,
};

fn usize_values(values: &[i64], name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| PyValueError::new_err(format!("{name} must be non-negative")))
        })
        .collect()
}

fn validate_equal_lengths(left: usize, right: usize, names: &str) -> PyResult<()> {
    if left != right {
        return Err(PyValueError::new_err(format!(
            "{names} must have the same length"
        )));
    }
    Ok(())
}

/// A road (or rail) network prepared once (contraction hierarchy) and reused
/// for Arrow-native snapping and routing queries.
#[pyclass]
pub struct RoadNetworkHandle {
    graph: RoadGraph,
}

#[pymethods]
impl RoadNetworkHandle {
    #[new]
    fn new(
        edge_from: PyArray,
        edge_to: PyArray,
        edge_weight_ds: PyArray,
        edge_length_m: PyArray,
        node_lat: PyArray,
        node_lng: PyArray,
    ) -> PyResult<Self> {
        let edge_from = as_i64_array(edge_from, "edge_from")?;
        let edge_to = as_i64_array(edge_to, "edge_to")?;
        let edge_weight_ds = as_i64_array(edge_weight_ds, "edge_weight_ds")?;
        let edge_length_m = as_f64_array(edge_length_m, "edge_length_m")?;
        let node_lat = as_f64_array(node_lat, "node_lat")?;
        let node_lng = as_f64_array(node_lng, "node_lng")?;
        validate_equal_lengths(edge_from.len(), edge_to.len(), "edge_from and edge_to")?;
        validate_equal_lengths(edge_from.len(), edge_weight_ds.len(), "edge arrays")?;
        validate_equal_lengths(edge_from.len(), edge_length_m.len(), "edge arrays")?;
        validate_equal_lengths(node_lat.len(), node_lng.len(), "node_lat and node_lng")?;
        let edge_from = usize_values(arrow_i64_values(&edge_from), "edge_from")?;
        let edge_to = usize_values(arrow_i64_values(&edge_to), "edge_to")?;
        let edge_weight_ds = usize_values(arrow_i64_values(&edge_weight_ds), "edge_weight_ds")?;
        Ok(Self {
            graph: RoadGraph::build_with_length_and_coords(
                &edge_from,
                &edge_to,
                &edge_weight_ds,
                arrow_values(&edge_length_m),
                arrow_values(&node_lat),
                arrow_values(&node_lng),
            ),
        })
    }

    fn batch_nearest_nodes(
        &self,
        py: Python<'_>,
        latitudes: PyArray,
        longitudes: PyArray,
        max_distance_m: f64,
    ) -> PyResult<(PyArray, PyArray)> {
        let latitudes = as_f64_array(latitudes, "latitudes")?;
        let longitudes = as_f64_array(longitudes, "longitudes")?;
        validate_equal_lengths(
            latitudes.len(),
            longitudes.len(),
            "latitudes and longitudes",
        )?;
        let (nodes, distances) = py.detach(|| {
            self.graph
                .batch_nearest_nodes(arrow_values(&latitudes), arrow_values(&longitudes))
        });
        let nodes: Vec<i64> = nodes
            .into_iter()
            .zip(distances.iter())
            .map(|(node, distance)| {
                if *distance <= max_distance_m {
                    node
                } else {
                    -1
                }
            })
            .collect();
        Ok((
            i64_results_into_arrow(nodes),
            f64_results_into_arrow(distances),
        ))
    }

    fn batch_distances(
        &self,
        py: Python<'_>,
        from_nodes: PyArray,
        to_nodes: PyArray,
    ) -> PyResult<(PyArray, PyArray)> {
        let from_nodes = as_i64_array(from_nodes, "from_nodes")?;
        let to_nodes = as_i64_array(to_nodes, "to_nodes")?;
        validate_equal_lengths(from_nodes.len(), to_nodes.len(), "from_nodes and to_nodes")?;
        let (distances, connected) = py.detach(|| {
            batch_road_distances(
                &self.graph,
                arrow_i64_values(&from_nodes),
                arrow_i64_values(&to_nodes),
            )
        });
        Ok((
            f64_results_into_arrow(distances),
            bool_results_into_arrow(connected),
        ))
    }

    #[allow(clippy::type_complexity)]
    fn batch_routes(
        &self,
        py: Python<'_>,
        from_nodes: PyArray,
        to_nodes: PyArray,
        max_waypoints: usize,
    ) -> PyResult<(PyArray, PyArray, PyArray, PyArray, PyArray, PyArray)> {
        let from_nodes = as_i64_array(from_nodes, "from_nodes")?;
        let to_nodes = as_i64_array(to_nodes, "to_nodes")?;
        validate_equal_lengths(from_nodes.len(), to_nodes.len(), "from_nodes and to_nodes")?;
        let (lats, lngs, weights, connected, starts, ends) = py.detach(|| {
            batch_road_routes(
                &self.graph,
                arrow_i64_values(&from_nodes),
                arrow_i64_values(&to_nodes),
                max_waypoints,
            )
        });
        Ok((
            f64_results_into_arrow(lats),
            f64_results_into_arrow(lngs),
            i64_results_into_arrow(weights),
            bool_results_into_arrow(connected),
            u64_results_into_arrow(starts.into_iter().map(|v| v as u64).collect()),
            u64_results_into_arrow(ends.into_iter().map(|v| v as u64).collect()),
        ))
    }

    fn route_edge_flows(
        &self,
        py: Python<'_>,
        from_nodes: PyArray,
        to_nodes: PyArray,
        flows: PyArray,
    ) -> PyResult<(PyArray, PyArray, PyArray, f64)> {
        let from_nodes = as_i64_array(from_nodes, "from_nodes")?;
        let to_nodes = as_i64_array(to_nodes, "to_nodes")?;
        let flows = as_f64_array(flows, "flows")?;
        validate_equal_lengths(from_nodes.len(), to_nodes.len(), "from_nodes and to_nodes")?;
        validate_equal_lengths(from_nodes.len(), flows.len(), "node and flow arrays")?;
        let (from, to, total, dropped) = py.detach(|| {
            batch_route_edge_flows(
                &self.graph,
                arrow_i64_values(&from_nodes),
                arrow_i64_values(&to_nodes),
                arrow_values(&flows),
            )
        });
        Ok((
            u64_results_into_arrow(from.into_iter().map(|v| v as u64).collect()),
            u64_results_into_arrow(to.into_iter().map(|v| v as u64).collect()),
            f64_results_into_arrow(total),
            dropped,
        ))
    }
}

#[pyfunction]
#[pyo3(name = "subsample_waypoints")]
pub fn subsample_waypoints_arrow(
    lats: PyArray,
    lngs: PyArray,
    times: PyArray,
    max_points: usize,
) -> PyResult<(PyArray, PyArray, PyArray)> {
    let lats = as_f64_array(lats, "lats")?;
    let lngs = as_f64_array(lngs, "lngs")?;
    let times = as_i64_array(times, "times")?;
    validate_equal_lengths(lats.len(), lngs.len(), "lats and lngs")?;
    validate_equal_lengths(lats.len(), times.len(), "coordinates and times")?;
    let (out_lats, out_lngs, out_times) = subsample_waypoints(
        arrow_values(&lats),
        arrow_values(&lngs),
        arrow_i64_values(&times),
        max_points,
    );
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        i64_results_into_arrow(out_times),
    ))
}

#[pyfunction]
pub fn nearest_nodes_arrow(
    py: Python<'_>,
    query_lat: PyArray,
    query_lng: PyArray,
    reference_lat: PyArray,
    reference_lng: PyArray,
    max_distance_m: f64,
) -> PyResult<(PyArray, PyArray)> {
    let query_lat = as_f64_array(query_lat, "query_lat")?;
    let query_lng = as_f64_array(query_lng, "query_lng")?;
    let reference_lat = as_f64_array(reference_lat, "reference_lat")?;
    let reference_lng = as_f64_array(reference_lng, "reference_lng")?;
    validate_equal_lengths(query_lat.len(), query_lng.len(), "query coordinates")?;
    validate_equal_lengths(
        reference_lat.len(),
        reference_lng.len(),
        "reference coordinates",
    )?;
    let (nodes, distances) = py.detach(|| {
        batch_nearest_coordinates(
            arrow_values(&query_lat),
            arrow_values(&query_lng),
            arrow_values(&reference_lat),
            arrow_values(&reference_lng),
        )
    });
    let nodes = nodes
        .into_iter()
        .zip(distances.iter())
        .map(
            |(node, &distance)| {
                if distance <= max_distance_m {
                    node
                } else {
                    -1
                }
            },
        )
        .collect();
    Ok((
        i64_results_into_arrow(nodes),
        f64_results_into_arrow(distances),
    ))
}
