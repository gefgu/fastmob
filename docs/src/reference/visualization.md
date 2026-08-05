# Visualization

Install the optional visualisation package with `pip install fastmob[vis]`.
Its documented entry point is `fastmob.vis`; the wheel is named `fastmob-vis`
and its direct Python import is `fastmob_vis`.

`fastmob.vis` accepts any Narwhals-compatible eager dataframe. Chart constructors
use explicit field mappings, so one API works for pandas and Polars results.
They return immutable, notebook-renderable `Chart` specifications.

```python
import fastmob
from fastmob import vis

jumps = fastmob.jump_lengths(trajectory)
chart = vis.ecdf(jumps, value_col="jump_lengths", x_label="jump length", x_unit="km")
chart.to_html("jump-lengths.html")
```

The same call works with pandas and Polars because the chart receives the
measure result through Narwhals:

```python
import polars as pl

result = pl.DataFrame({"jump_lengths": [[0.8, 1.2], [2.4]]})
vis.ecdf(result, value_col="jump_lengths")
```

```python
import pandas as pd

result = pd.DataFrame({"jump_lengths": [[0.8, 1.2], [2.4]]})
vis.ecdf(result, value_col="jump_lengths")
```

Compare two results by providing `second`; the default series names are `first`
and `second` and can be overridden with `labels`.

```python
chart = vis.ecdf(first_jumps, value_col="jump_lengths", second=second_jumps,
                 labels=("observed", "simulation"))
```

## Generic charts

- `ecdf(data, value_col=...)` accepts scalar or list-valued numeric measure columns.
- `bar(data, category_col=..., value_col=..., series_col=...)` draws categorical results.
- `heatmap(data, x_col=..., y_col=..., value_col=...)` draws long-form matrices.
- `scatter(data, x_col=..., y_col=..., series_col=...)` overlays measure or fit results.
- `boxplot(data, category_col=..., value_col=..., series_col=...)` summarizes long-form values.

Every Fastmob dataframe wrapper exposes matching `plot_ecdf`, `plot_bar`,
`plot_heatmap`, `plot_scatter`, and `plot_boxplot` helpers for its existing
columns. They do not recompute measures.

## Export and embedding

`Chart.to_dict()` returns the ECharts option for web APIs. `to_json`, `to_html`,
and `to_svg` write portable artifacts; `render()` returns `EChartsFigure` for
low-level control. Use `get_resource_bundle()` and `bundle_libs=False` when
embedding several charts in one document.

The former Folium/Matplotlib plotting API was removed in the `fastmob-vis`
transition. Use the generic ECharts charts or a dedicated GIS renderer instead.
