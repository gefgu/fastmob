# About Fastmob

Fastmob is a high-performance, backend-agnostic Python library for mobility
analysis. It combines a familiar dataframe-facing API with Rust kernels for
the compute-heavy parts of trajectory processing and measurement.

## What it supports

- eager dataframe inputs through Narwhals, including pandas, Polars, and
  PyArrow-compatible workflows;
- a trajectory hierarchy from raw position fixes through stays, trip legs,
  trips, tours, and flows;
- individual and collective mobility measures, statistical fitting, models,
  privacy assessment, and optional road/rail-network analysis.

## Project status

Fastmob is under active development. Public behavior, supported Python
versions, optional extras, and migration notes are documented in the current
[reference](../reference/index.md) and [release notes](../release_notes/index.md).
Pin a released version for reproducible research or production pipelines.

## Contribute and report issues

- [Source code and issue tracker](https://github.com/gefgu/fastmob)
- [Release history](../release_notes/index.md)
- [Documentation recipes](../learn/index.md)

When reporting an issue, include Fastmob and Python versions, dataframe
backend/version, a minimal input, expected output, and the complete error.
