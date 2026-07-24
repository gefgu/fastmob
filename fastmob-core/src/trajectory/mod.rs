//! Trajectory interpolation and trajectory-pair similarity/distance kernels.
//!
//! Backs the Python `fastmob.trajectory` namespace: bulk gap-filling
//! (`interpolate`), point-in-time position queries (`interpolate_at`), and
//! pairwise shape-comparison metrics (`trajectory_distance`).

pub mod distance;
pub mod interpolate;
pub mod interpolate_at;
pub mod smooth;
