use polars::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastmobRsError {
    #[error("could not find required column(s): {0}. Available columns: {1:?}")]
    MissingColumns(String, Vec<String>),
    #[error("column {0:?} has unsupported dtype {1:?}")]
    UnsupportedDtype(String, DataType),
    #[error("fastmob-core error: {0}")]
    Core(String),
    #[error("fastmob-rs invariant violated: {0}")]
    Invariant(String),
    #[error(transparent)]
    Polars(#[from] PolarsError),
}
