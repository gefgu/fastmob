# Tessellation

Tilers that cover an area with square, hexagonal (H3) or Voronoi cells and return a GeoPandas table with a
`tile_id` column, ready for `TrajDataFrame.mapping` and `to_flowdataframe`. They need the optional
`geopandas` and `shapely` packages. Tilers are created through the `tilers.tiler` registry, for example
`tilers.tiler.create("squared", base_shape=area, meters=500)`; see
[Fill gaps, smooth, compare and convert trajectories](../learn/process-trajectories.md) for a worked example.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`TessellationTilers`](#fastmob.tessellation.tilers.TessellationTilers) | Registry for tessellation tiler implementations. |  |
| [`create`](#fastmob.tessellation.tilers.TessellationTilers.create) | Build a tessellation with the tiler registered under `key` (`squared`, `h3_tessellation`, `voronoi` or your own). | [Build a grid and snap positions to it](../learn/process-trajectories.md#build-a-grid-and-snap-positions-to-it) |
| [`get`](#fastmob.tessellation.tilers.TessellationTilers.get) | Alias of `create`: build a tessellation with the tiler registered as `service_id`. | [Build a grid and snap positions to it](../learn/process-trajectories.md#build-a-grid-and-snap-positions-to-it) |
| [`register_tiler`](#fastmob.tessellation.tilers.TessellationTilers.register_tiler) | Register `tiler_instance` (a callable tiler object) under the name `key`, replacing any tiler of that name. | [Build a grid and snap positions to it](../learn/process-trajectories.md#build-a-grid-and-snap-positions-to-it) |
| [`SquaredTessellationTiler`](#fastmob.tessellation.tilers.SquaredTessellationTiler) | Build a square grid over a base shape. |  |
| [`H3TessellationTiler`](#fastmob.tessellation.tilers.H3TessellationTiler) | Build an H3 hexagonal tessellation over a base shape. |  |
| [`VoronoiTessellationTiler`](#fastmob.tessellation.tilers.VoronoiTessellationTiler) | Build a point tessellation from input points. |  |
<!-- api-table:end -->

::: fastmob.tessellation.tilers.TessellationTilers
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Build a grid and snap positions to it](../learn/process-trajectories.md#build-a-grid-and-snap-positions-to-it)

---

::: fastmob.tessellation.tilers.SquaredTessellationTiler
    options:
      show_source: false

---

::: fastmob.tessellation.tilers.H3TessellationTiler
    options:
      show_source: false

---

::: fastmob.tessellation.tilers.VoronoiTessellationTiler
    options:
      show_source: false
