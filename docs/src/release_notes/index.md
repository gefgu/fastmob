# Release notes

!!! tip "Latest release: v0.3.2 (2026-09-19)"

    `pip install -U fastmob`. No breaking changes in this release. Pin
    `fastmob==X.Y.Z` for reproducible research.

| Version | Highlights |
| --- | --- |
| [v0.3.2](v0.3.2.md)<br><small>2026-09-19</small> | Stable `fastmob.vis` facade; `fastmob[vis]` pins a compatible `fastmob-vis` 0.1.x. |
| [v0.3.1](v0.3.1.md)<br><small>2026-09-18</small> | Docs only: contributing guide and GeoLife benchmark story. |
| [v0.3.0](v0.3.0.md)<br><small>2026-09-18</small> | [Spatiotemporal visitation comparison](../reference/measures/evaluation.md) scales to 100M-row inputs; contact-network analysis shares one faster set of kernels. |
| [v0.2.2](v0.2.2.md)<br><small>2026-09-09</small> | Better staypoint detection; packaging improvements. |
| [v0.2.1](v0.2.1.md)<br><small>2026-08-25</small> | Faster stay-location detection and GeoLife loading; fastmob dataframes accepted by every public API. |
| [v0.2.0](v0.2.0.md)<br><small>2026-08-23</small> | [Trajectory hierarchy](../reference/data-structures.md), new [preprocessing](../reference/preprocessing.md) algorithms, [models](../reference/models.md), and network and social analysis. |
| [v0.1.0 to v0.1.3](v0.1.0.md)<br><small>2026-07-16 to 07-21</small> | Initial public release followed by maintenance releases ([v0.1.1](v0.1.1.md), [v0.1.2](v0.1.2.md), [v0.1.3](v0.1.3.md)). |

## Upgrade policy

- fastmob is pre-1.0: a minor version bump (0.x to 0.y) may change behavior.
- Breaking changes and migrations are called out on the matching release page.
- `fastmob[vis]` installs a `fastmob-vis` release from the compatible 0.1.x line.
- Pin `fastmob==X.Y.Z` for reproducible research.

Upgrade trouble? [Open an issue](https://github.com/gefgu/fastmob/issues) or see all [GitHub releases](https://github.com/gefgu/fastmob/releases).
