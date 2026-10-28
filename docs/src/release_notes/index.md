# Release notes

!!! tip "Latest release: v0.3.4 (2026-10-28)"

    `pip install -U fastmob`. No breaking changes in this release. Pin
    `fastmob==X.Y.Z` for reproducible research.

| Version | Highlights |
| --- | --- |
| [v0.3.4](v0.3.4.md)<br><small>2026-10-28</small> | Fixes a wrong [`common_part_of_commuters_distance`](../reference/measures/evaluation.md) result, `load_geolife` caching and `interpolate_at` with pandas time indexes; 18 runnable Learn notebooks, a chart gallery and the full `fastmob-vis` reference. |
| [v0.3.3](v0.3.3.md)<br><small>2026-10-27</small> | [`load_geolife` and `load_brightkite`](../reference/data.md) download just the part of a public dataset you ask for. |
| [v0.3.2](v0.3.2.md)<br><small>2026-09-19</small> | Stable `fastmob.vis` facade; `fastmob[vis]` pins a compatible `fastmob-vis` 0.1.x. |
| [v0.3.1](v0.3.1.md)<br><small>2026-09-18</small> | Docs only: contributing guide and GeoLife benchmark story. |
| [v0.3.0](v0.3.0.md)<br><small>2026-09-18</small> | [Spatiotemporal visitation comparison](../reference/measures/evaluation.md) scales to 100M-row inputs; contact-network analysis shares one set of kernels. |
| [v0.2.2](v0.2.2.md)<br><small>2026-09-09</small> | Better staypoint detection; packaging improvements. |
| [v0.2.1](v0.2.1.md)<br><small>2026-08-25</small> | Optimized stay-location detection and GeoLife loading; Fastkit-Mobility dataframes accepted by every public API. |
| [v0.2.0](v0.2.0.md)<br><small>2026-08-23</small> | [Trajectory hierarchy](../reference/data-structures.md), new [preprocessing](../reference/preprocessing.md) algorithms, [models](../reference/models.md), and network and social analysis. |
| [v0.1.0 to v0.1.3](v0.1.0.md)<br><small>2026-07-16 to 07-21</small> | Initial public release followed by maintenance releases ([v0.1.1](v0.1.1.md), [v0.1.2](v0.1.2.md), [v0.1.3](v0.1.3.md)). |

## Upgrade policy

- Fastkit-Mobility is pre-1.0: a minor version bump (0.x to 0.y) may change behavior.
- Breaking changes and migrations are called out on the matching release page.
- `fastmob[vis]` installs a `fastmob-vis` release from the compatible 0.1.x line.
- Pin `fastmob==X.Y.Z` for reproducible research.

Upgrade trouble? [Open an issue](https://github.com/gefgu/fastmob/issues) or see all [GitHub releases](https://github.com/gefgu/fastmob/releases).
