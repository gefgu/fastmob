# Key Features

- **Backend-agnostic**: pass a pandas, polars, or any other Narwhals-compatible DataFrame — fkmob works without changes.

- **Rust-accelerated core**: 500x median speedup due to rust parallelized and zero-copy operations.
    
- **Drop-in API**: function signatures mirror the original skmob library so migration is straightforward.
    
- **Measured validation**: correctness tests, Python coverage, profiling, and standalone speed benchmarks make compatibility and performance claims reproducible.

- **Zero-Copy**: Scikit-Mobility 2 process the data where it lives. Instead of copying, it directly access your dataframe in the memory, saving memory and making the processing faster.

- **Lightweight**: Python Wheels ship with less than 30KB. 