---
icon: lucide/chart-no-axes-combined
---

# FlowDataFrame

`FlowDataFrame` is fastmob's sparse origin-destination wrapper. It represents
an aggregate flow table independently of the trajectory hierarchy, while a
`Trips` collection with global endpoint locations can create one directly.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`FlowDataFrame`](#fastmob.core.flow_dataframe.FlowDataFrame) | Narwhals-backed wrapper for origin-destination flow data. |  |
| [`common_part_of_commuters`](#fastmob.core.flow_dataframe.FlowDataFrame.common_part_of_commuters) | Compare sparse OD flows with another FlowDataFrame using Rust CPC. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`common_part_of_links`](#fastmob.core.flow_dataframe.FlowDataFrame.common_part_of_links) | Return the common part of links score against another flow table. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`get_flow`](#fastmob.core.flow_dataframe.FlowDataFrame.get_flow) | Return the flow between two tile IDs (0 if no such pair exists). | [Flows between zones](../../learn/population-measures.md#flows-between-zones) |
| [`get_geometry`](#fastmob.core.flow_dataframe.FlowDataFrame.get_geometry) | Return the geometry of a tessellation tile. | [Flows between zones](../../learn/population-measures.md#flows-between-zones) |
| [`settings_from`](#fastmob.core.flow_dataframe.FlowDataFrame.settings_from) | Copy metadata attributes from another FlowDataFrame. | [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) |
| [`to_matrix`](#fastmob.core.flow_dataframe.FlowDataFrame.to_matrix) | Convert the flow table to a numpy matrix. | [Flows between zones](../../learn/population-measures.md#flows-between-zones) |
<!-- api-table:end -->

## Required data

The table needs origin, destination, and flow-value columns. A tessellation or
location geometry table is optional and enables geometry lookup and legacy
mapping helpers. Column names can be supplied explicitly when constructing the
wrapper.

```python
from fastmob import FlowDataFrame

flows = FlowDataFrame({
    "origin": ["A", "A"],
    "destination": ["B", "C"],
    "flow": [12, 4],
})

between_a_and_b = flows.get_flow("A", "B")
matrix = flows.to_matrix()
```

For trajectory-derived flows, start from [`Trips`](trips.md) and call
`to_flow_dataframe()` after assigning global location IDs.

## API

::: fastmob.core.flow_dataframe.FlowDataFrame
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Compare two flow tables](../../learn/population-measures.md#compare-two-flow-tables) · [Flows between zones](../../learn/population-measures.md#flows-between-zones)

