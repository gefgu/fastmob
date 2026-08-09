---
icon: lucide/chart-no-axes-combined
---

# FlowDataFrame

`FlowDataFrame` is fastmob's sparse origin-destination wrapper. It represents
an aggregate flow table independently of the trajectory hierarchy, while a
`Trips` collection with global endpoint locations can create one directly.

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
