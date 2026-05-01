# Backend-agnostic, Rust-accelerated design

skmob2 is designed around two goals that often pull in different directions: accepting the dataframe tools people already use, and making compute-heavy mobility measures fast enough for large trajectories.

The library resolves that tension by using Narwhals at the Python boundary and Rust for selected numerical kernels. Narwhals gives skmob2 a common dataframe interface across eager backends such as pandas and Polars. Rust gives the implementation a compiled path for repeated geographic calculations, grouping work, and array-oriented routines.

## Why dataframe compatibility matters

Mobility analysis often starts after data has already passed through another workflow. One project may use pandas because it is familiar and widely supported. Another may use Polars because the input is larger or because the surrounding pipeline is already Polars-based.

Making users convert dataframes before every measure would add friction and can hide accidental changes in types, null handling, or column order. The backend-agnostic approach keeps skmob2 closer to the user's existing workflow: pass in an eager dataframe, receive the same backend where a dataframe result is returned.

This is useful because skmob2 can focus on mobility semantics while the caller keeps control of their dataframe ecosystem.

## Why Rust is used for kernels

Many mobility measures perform the same low-level operations repeatedly: compute Haversine distances, scan consecutive points, group records by user, or aggregate numeric arrays. These operations are small individually, but they become expensive across millions of trajectory points.

Rust is a good fit for that layer because it provides predictable compiled performance while still integrating with Python through PyO3. In practice, this means the Python API can stay familiar while the inner loops avoid much of the overhead that pure Python implementations would pay.

The result is not that every part of skmob2 needs to be Rust. The useful split is more precise: Python coordinates dataframe-facing behavior and public API compatibility; Rust handles the dense numerical work where compiled code matters most.

## How the layers fit together

The Python layer receives a dataframe, detects or accepts the important trajectory columns, prepares the data, and preserves the calling convention users see in the [API reference](../api/index.md). Once the relevant columns are ready, selected measures route compact arrays into Rust kernels.

Narwhals is the translation layer that makes this practical. It gives skmob2 enough shared dataframe behavior to prepare inputs without committing the public API to one dataframe library. The Rust kernels then work on the numerical representation rather than on pandas-specific or Polars-specific objects.

This division keeps the public surface small: users call functions such as `jump_lengths` or `radius_of_gyration`, not a separate pandas API and Polars API.

## Tradeoffs

This approach trades some implementation simplicity for a friendlier user surface. Supporting multiple dataframe backends requires careful handling of data types, nulls, sorting, and output reconstruction. It is more work than writing only for pandas.

The benefit is that the same mobility API can fit into more workflows. Users who value familiarity can stay with pandas. Users who value Polars' execution model can keep Polars at the edges of their analysis. skmob2 carries the compatibility burden so each project does not have to.

Rust introduces a similar tradeoff. It adds a compiled extension and a build toolchain for source installs, but it gives the project a clear place to optimize the routines that dominate runtime. For installation and development details, see [Getting Started](../getting-started.md).

## What this means in practice

The design aims to make the fast path feel ordinary. A trajectory measure should look like a normal Python function call, accept the dataframe already in the user's hands, and return a result that fits back into the same analysis environment.

That is the central idea behind skmob2: keep the mobility-analysis API familiar, let dataframe choice remain flexible, and reserve lower-level optimization for the places where it changes real workloads.

