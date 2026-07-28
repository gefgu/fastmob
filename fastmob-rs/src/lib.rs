//! Public Rust DataFrame API for fastmob mobility analysis.
//!
//! `fastmob-core` holds the compute kernels; this crate is the Polars
//! orchestration layer around them, so Rust callers get the same measures the
//! Python package exposes without reimplementing column detection, cleaning,
//! arranging and group-boundary bookkeeping at every call site.
//!
//! The crate is **presorted-only**: [`prepare`] cleans and arranges a frame
//! once, and every measure is a pure kernel over the resulting
//! [`PreparedTrajectory`].
//!
//! ```no_run
//! use fastmob_rs::{Cols, jump_lengths_flat, prepare, radius_of_gyration_flat};
//! # fn main() -> Result<(), fastmob_rs::FastmobRsError> {
//! # let df = polars::prelude::DataFrame::empty();
//! let prep = prepare(&df, Cols::auto())?;
//! let jumps = jump_lengths_flat(&prep)?;
//! let rog = radius_of_gyration_flat(&prep);
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod measures;
pub mod prepare;

pub use error::FastmobRsError;
pub use measures::individual::{
    jump_lengths, jump_lengths_flat, radius_of_gyration, radius_of_gyration_flat,
};
pub use prepare::{Cols, PreparedTrajectory, ResolvedColumns, prepare, prepare_keeping};
