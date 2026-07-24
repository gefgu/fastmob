# fastmob-vis

Rust-backed ECharts/Leaflet visualizations for [fastmob](https://github.com/gefgu/fastmob)
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
from fastmob_vis import plot_jump_lengths_ecdf

fig = plot_jump_lengths_ecdf([2.0, 1.0, 1.0], [3.0, 1.5], labels=("observed", "synthetic"))
fig
```

Or through the main `fastmob` package (requires `fastmob[vis]` to be installed):

```python
from fastmob import vis

fig = vis.plot_jump_lengths_ecdf([2.0, 1.0, 1.0], [3.0, 1.5], labels=("observed", "synthetic"))
fig
```

In Jupyter, the returned figure renders with ECharts. For web servers, use
`fig.to_dict()`. To write full, non-truncated artifacts, use
`fig.to_json("chart.json")` or `fig.to_html("chart.html")`.

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
