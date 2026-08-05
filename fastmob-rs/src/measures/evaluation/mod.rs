pub mod jsd;
pub mod wasserstein;

pub use jsd::{jensen_shannon_divergence, time_bin_matrix_jsd};
pub use wasserstein::wasserstein_distance;
