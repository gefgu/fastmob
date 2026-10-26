# Release notes

!!! tip "Latest release: v0.3.2 (2026-09-19)"

    `pip install -U fastmob`. No breaking changes in this release. Pin
    `fastmob==X.Y.Z` for reproducible research.

| Version | Released | Highlights |
| --- | --- | --- |
| [v0.3.2](v0.3.2.md) | 2026-09-19 | Stable `fastmob.vis` facade; `fastmob[vis]` pins a compatible `fastmob-vis` 0.1.x. |
| [v0.3.1](v0.3.1.md) | 2026-09-18 | Docs only: contributing guide and GeoLife benchmark story. |
| [v0.3.0](v0.3.0.md) | 2026-09-18 | Scalable STVD with a sparse EMD path, shared RECAST contact-network kernels. |
| [v0.2.2](v0.2.2.md) | 2026-09-09 | Better staypoint detection; packaging improvements. |
| [v0.2.1](v0.2.1.md) | 2026-08-25 | Faster `stay_locations` and GeoLife loading; wrapper dataframes accepted everywhere. |
| [v0.2.0](v0.2.0.md) | 2026-08-23 | [Trajectory hierarchy](../reference/data-structures.md), new [preprocessing](../reference/preprocessing.md) algorithms, [models](../reference/models.md), network and social analysis, Arrow-only Rust bindings. |

??? note "0.1.x series (2026-07-16 to 2026-07-21)"

    [v0.1.3](v0.1.3.md), [v0.1.2](v0.1.2.md) and [v0.1.1](v0.1.1.md) were
    maintenance releases after the initial public release,
    [v0.1.0](v0.1.0.md).

## Upgrade policy

fastmob is pre-1.0: a minor version bump may change behavior. Breaking
changes and migrations are called out in the matching release page.
