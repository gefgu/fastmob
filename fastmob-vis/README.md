# fastmob-vis

Narwhals-native ECharts visualizations for [fastmob](https://github.com/gefgu/fastmob)
mobility-analysis results.

This is an optional add-on to `fastmob`: it lives in the same repository and
Cargo workspace, but ships as its own wheel so that a plain `pip install
fastmob` never has to build or depend on it.

## Installation

```bash
pip install fastmob[vis]
```

or standalone:

```bash
pip install fastmob-vis
```

For development from source (as a workspace member of `fastmob`):

```bash
cd fastmob-vis
uv sync --extra dev
env -u CONDA_PREFIX uv run maturin develop
```

## Example

Directly:

```python
import pyarrow as pa
from fastmob import vis

first = pa.table({"jump_lengths": [[0.8, 1.2], [2.4]]})
second = pa.table({"jump_lengths": [[0.7, 1.4], [2.1]]})
chart = vis.ecdf(first, value_col="jump_lengths", second=second)
chart
```

Generic constructors are available as `ecdf`, `bar`, `heatmap`, `scatter`, and
`boxplot`. They accept any Narwhals-compatible eager dataframe and explicit
field mappings. The result is an immutable `Chart`, which can be rendered or
exported:

```python
chart.to_html("chart.html")
chart.to_svg("chart.svg")
```

In Jupyter, `Chart` renders with ECharts. For web servers, use `chart.to_dict()`.
`Chart.render()` returns the lower-level `EChartsFigure` when needed.

`fastmob-vis` is a clean break from its former legacy API: Folium and
Matplotlib maps, their dependencies, and the old measure-specific `plot_*`
functions are not provided.

## Development Checks

```bash
uv run pytest
uv run ruff check
```

## License

`fastmob-vis` is released under the same [BSD-3-Clause License](../LICENSE)
as the rest of the `fastmob` repository. It bundles a few third-party JS/CSS
libraries for standalone chart rendering — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for their licenses.
