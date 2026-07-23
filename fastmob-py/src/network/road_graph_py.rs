use fastmob_core::network::road_graph::{RoadGraph, batch_road_distances, subsample_waypoints};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn i64_as_usize_vec(arr: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    Ok(arr.as_slice()?.iter().map(|&x| x.max(0) as usize).collect())
}

/// A road (or rail) network prepared once (contraction hierarchy) and
/// reused for many point-to-point physical-distance queries -- CH
/// preparation, not the query itself, is the expensive step, so one handle
/// serves every query batch needed by a session instead of re-preparing per
/// call.
#[pyclass]
pub struct RoadNetworkHandle {
    graph: RoadGraph,
}

#[pymethods]
impl RoadNetworkHandle {
    #[new]
    fn new(
        edge_from: PyReadonlyArray1<'_, i64>,
        edge_to: PyReadonlyArray1<'_, i64>,
        edge_weight_ds: PyReadonlyArray1<'_, i64>,
        edge_length_m: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let ef = i64_as_usize_vec(&edge_from)?;
        let et = i64_as_usize_vec(&edge_to)?;
        let ew = i64_as_usize_vec(&edge_weight_ds)?;
        let el = edge_length_m.as_slice()?;
        Ok(Self {
            graph: RoadGraph::build_with_length(&ef, &et, &ew, el),
        })
    }

    /// Batch physical-distance (metres) query for `(from_node, to_node)`
    /// pairs against the prepared contraction hierarchy. Returns
    /// `(distances_m, connected)`, `connected` as `0`/`1` per pair. The
    /// Python caller falls back to straight-line Haversine wherever
    /// `connected == 0` (negative/unsnapped node ids or a disconnected
    /// graph component).
    fn batch_distances<'py>(
        &self,
        py: Python<'py>,
        from_nodes: PyReadonlyArray1<'py, i64>,
        to_nodes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>)> {
        let from_slice = from_nodes.as_slice()?;
        let to_slice = to_nodes.as_slice()?;
        let (dist, conn) = py.detach(|| batch_road_distances(&self.graph, from_slice, to_slice));
        let conn_u8: Vec<u8> = conn.into_iter().map(|b| b as u8).collect();
        Ok((dist.into_pyarray(py), conn_u8.into_pyarray(py)))
    }
}

/// Standalone binding for `subsample_waypoints`, for callers (e.g. a
/// simulation's per-trip route logging) that don't otherwise need a full
/// `RoadNetworkHandle`.
#[pyfunction]
#[pyo3(name = "subsample_waypoints")]
pub fn subsample_waypoints_numpy<'py>(
    py: Python<'py>,
    lats: PyReadonlyArray1<'py, f64>,
    lngs: PyReadonlyArray1<'py, f64>,
    times: PyReadonlyArray1<'py, i64>,
    max_points: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = lats.as_slice()?;
    let lngs = lngs.as_slice()?;
    let times = times.as_slice()?;
    let (out_lats, out_lngs, out_times) = subsample_waypoints(lats, lngs, times, max_points);
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_times.into_pyarray(py),
    ))
}
