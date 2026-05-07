# Correctness, coverage, and benchmarking

skmob2 is a reimplementation of a familiar mobility-analysis API, so performance alone is not enough. The library has to be faster while still preserving mobility semantics, dataframe behavior, and compatibility with results users already trust.

The validation approach reflects that tension. Tests check known examples and edge cases, coverage shows which Python paths are exercised, and benchmarks measure the workloads where the Rust-accelerated design is expected to matter.

## Why validation is layered

Mobility measures combine several kinds of behavior. Some parts are mathematical, such as Haversine distances, entropy calculations, or fitting routines. Other parts are about data handling: column detection, optional user identifiers, datetime sorting, null handling, and preserving the caller's dataframe backend.

A single test style would miss part of that picture. Synthetic correctness tests are useful because they make small expectations explicit. They can pin down edge cases such as one-point trajectories, missing optional columns, tied ranks, or backend-specific output details.

Comparison tests serve a different role. When a skmob2 function mirrors an existing skmob measure, the comparison suite checks skmob2 against skmob on shared inputs, including Brightkite-derived trajectories where the optional dependency is installed. This is useful because API compatibility is not only a matter of names and signatures; it also includes result shapes, tolerances, and conventions that users notice during migration.

The Rust boundary needs its own attention. Python tests exercise public functions that route arrays into compiled kernels, while lower-level checks compare Rust-backed paths with Python-facing expectations. That combination keeps the fast path connected to the behavior users call from Python.

## What coverage means here

The documented coverage metric is Python package coverage for `skmob2`. It measures which Python statements and branches run when the correctness suite executes:

```bash
bash tests/run_coverage.sh
```

This metric is intentionally scoped. It does not claim Rust line coverage for the PyO3 extension, and it does not turn performance benchmarks into correctness evidence. Instead, it answers a narrower question: how much of the Python API, dataframe preparation layer, and result reconstruction code is exercised by the correctness tests.

That distinction matters because skmob2 has two implementation layers. The Python layer owns the public API and backend adaptation. The Rust layer owns selected numerical kernels. Python coverage is the right signal for the first layer, while targeted correctness tests and benchmark workloads help validate the second.

## Why benchmarks are part of the design

skmob2 exists because mobility analysis can become expensive on large trajectory datasets. Benchmarks make that design pressure visible. They show whether compiled kernels and backend-agnostic data preparation improve real workloads, rather than only making individual functions look elegant in isolation.

The benchmark suite uses standalone `time.perf_counter()` scripts and includes Brightkite-sized slices where appropriate:

```bash
bash tests/run_benchmarks.sh
```

Some benchmarks compare skmob2 with skmob. Others exercise pandas and Polars inputs through the same public API. MovingPandas comparisons are included where the optional dependencies are installed and a comparable operation exists, because they answer a different question: how skmob2 behaves next to another trajectory-analysis ecosystem rather than only next to its direct predecessor. The benchmark runner dispatches those comparison groups to compatible virtual environments: `.venv` for skmob2 and movingpandas when installed there, and `.venv-skmob` for the legacy scikit-mobility stack.

Benchmark numbers should be read as measurements from a particular environment, dataset slice, dependency set, and implementation version. They are useful for tracking regressions and comparing approaches, but they are not permanent guarantees about every machine or every trajectory dataset.

## How the signals fit together

Correctness tests, coverage, profiling, and benchmarks answer different questions:

- correctness tests ask whether the result is right;
- coverage asks which Python paths were exercised while proving that;
- profiling asks where time and memory go;
- benchmarks ask whether performance changes hold under repeated measurement.

The value is in using those signals together. High coverage without comparison tests could still miss compatibility behavior. Fast benchmarks without correctness checks could reward the wrong result. Profiling without benchmarks could optimize code paths that are not important in practice.

In practice, skmob2 treats validation as part of the implementation, not as decoration around it. The public API stays familiar, the dataframe boundary stays flexible, and the performance story stays tied to reproducible tests that can be rerun as the library changes.
