#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! Low-level, allocation-conscious Rust kernels for mobility analysis.
//!
//! `fastmob-core` accepts primitive slices and returns Rust collections. It is
//! the compute layer used by the `fastmob` Python package and the
//! `fastmob-rs` DataFrame API, but it can also be used directly when callers
//! own their data layout.
//!
//! # Example
//!
//! ```
//! use fastmob_core::utils::haversine::haversine_km;
//!
//! let paris_to_london = haversine_km(48.8566, 2.3522, 51.5074, -0.1278);
//! assert!((paris_to_london - 343.0).abs() < 2.0);
//! ```
//!
//! # Compatibility
//!
//! The crate follows Cargo's pre-1.0 semantic-versioning conventions. Public
//! APIs are intended for direct use, but new capabilities and breaking changes
//! may be introduced in a minor release before version 1.0. The `*_impl`
//! entry points are low-level kernels and expose the same compatibility policy.
//!
//! See the crate README for supported feature flags, input conventions, and
//! third-party attribution notices.

pub mod hierarchy;
pub mod integration;
pub mod measures;
pub mod models;
pub mod network;
pub mod preprocessing;
pub mod privacy;
pub mod social;
pub mod trajectory;
pub mod utils;
