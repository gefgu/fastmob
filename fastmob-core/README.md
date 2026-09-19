# fastmob-core

Low-level Rust compute kernels for mobility and trajectory analysis.

`fastmob-core` powers the Python [`fastmob`](https://github.com/gefgu/fastmob)
package and the higher-level `fastmob-rs` DataFrame API. It is suitable for
Rust callers that own their columnar/slice-oriented data and want direct
access to trajectory preparation, measures, models, networks, privacy, and
social-analysis kernels.

## Installation

```toml
[dependencies]
fastmob-core = "0.2"
```

## Example

```rust
use fastmob_core::utils::haversine::haversine_km;

let distance_km = haversine_km(48.8566, 2.3522, 51.5074, -0.1278);
assert!((distance_km - 343.0).abs() < 2.0);
```

## Features

- `simd` (default): enables the accelerated `numkong` dependency and the
  `stvd-emd` kernels.
- `numkong`: enables only the optional acceleration dependency.
- `stvd-emd`: enables the STVD EMD and sparse Sinkhorn kernels.

Disable defaults when a portable, dependency-minimal build is preferred:

```toml
fastmob-core = { version = "0.2", default-features = false }
```

## API and compatibility

This crate exposes low-level kernels over primitive slices and collections.
Functions suffixed `_impl` are deliberate public entry points for bindings and
advanced Rust callers; they are not private implementation details. Before
1.0, compatibility follows Cargo's pre-1.0 semantic-versioning conventions,
so a minor release may include breaking changes.

## License and provenance

`fastmob-core` is BSD-3-Clause licensed. Some algorithms were ported or
adapted from other projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, license
status, and release requirements.
