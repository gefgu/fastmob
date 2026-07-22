# Rust Dispatcher Guidelines

These notes capture the binding-layer pattern introduced while refactoring
`radius_of_gyration`. Use them as the baseline when moving other metrics from
backend-specific PyO3 wrappers to shared Rust adapters.

## Core Kernels

- Keep `fastmob-core` backend-independent. Core functions should accept ordinary
  Rust slices and return ordinary Rust values.
- Do not introduce NumPy, Arrow, PyO3, or Narwhals types into core modules.
- Keep public API validation in the binding adapter. Core may assume validated
  lengths, ranges, indices, and null masks.
- Prefer explicit execution variants in core names. For grouped contiguous data,
  use `*_presorted_impl`; for indirect grouped data, use `*_indexed_impl`.
- Return the metadata the Python layer actually needs. For scalar-per-group
  metrics that may drop groups, prefer `(values, validity)` over counts when the
  Python layer only needs to know whether each group survives.

## PyO3 Binding Shape

- Expose logical execution variants, not backend variants. A metric should avoid
  separate `*_numpy`, `*_arrow`, `*_indexed_numpy`, and `*_indexed_arrow`
  functions when a shared adapter can dispatch by input type.
- Binding functions should accept generic Python array objects for backend-owned
  columns:
  - `&Bound<'py, PyAny>` for data columns that may be NumPy or Arrow.
  - Typed `PyReadonlyArray1<'py, usize>` for Rust-owned index and end arrays.
- Metric-specific PyO3 files should mostly wire arguments into an adapter and
  call the core kernel. They should not duplicate extraction, null handling,
  validation, GIL release, or output conversion.

## Adapter Responsibilities

- Detect backend family once:
  - all relevant inputs are NumPy arrays; or
  - all relevant inputs are Arrow arrays.
- Reject mixed or unsupported inputs with a clear `PyTypeError`.
- Extract borrowed structure-of-arrays views where possible. Avoid per-row
  objects or interleaved temporary coordinate structs.
- Validate shared invariants before releasing the GIL:
  - equal column lengths;
  - presorted end offsets are monotonic and in bounds;
  - indexed ends are monotonic and within the index array;
  - indices are within coordinate array bounds.
- Release the GIL around core execution with `py.detach`.
- Convert outputs back to the same backend family as the inputs for primary
  result values. Small side-channel masks or index arrays may remain NumPy when
  they are internal to the Python dispatch layer.

## Null And Validity Handling

- Centralize Arrow null handling in the adapter. Metric-specific binding files
  should not know how to combine Arrow validity buffers.
- For indexed coordinate metrics, build a combined `valid_rows: Option<&[bool]>`
  from nullable Arrow inputs and pass it to core.
- Preserve existing null semantics when refactoring. Do not make presorted Arrow
  paths nullable unless the metric intentionally changes that contract.
- If a group can become empty after null or finite-value filtering, return a
  validity mask aligned with the group output. Let Python filter values and user
  labels from that mask.

## Python Dispatch Layer

- Use `TrajectoryDispatcher` only where it still clarifies backend extraction or
  high-level dispatch. Do not keep a metric dispatcher only to route between
  backend-specific Rust functions that no longer exist.
- Keep dataframe column detection and common casting in shared helpers, not in
  metric-specific wrappers.
- Use shared result filtering helpers instead of metric-local Arrow/NumPy filter
  functions.
- If validity filtering drops users, emit one warning from the Python wrapper
  with the number of filtered users.

## Tests

- Remove tests for deleted backend-specific PyO3 functions.
- Add direct tests for the new logical PyO3 functions over both NumPy and Arrow
  inputs.
- Test mixed-backend rejection and validation errors at the adapter boundary.
- Test nullable Arrow indexed inputs when the metric supports null filtering.
- Test warning behavior when a validity mask filters out users.
