# Development setup

Install fastmob from source when you want to change the library or its Rust
kernels. Development needs [`uv`](https://docs.astral.sh/uv/) and a Rust
toolchain.

=== "Initial setup"

    ```bash
    git clone https://github.com/gefgu/fastmob.git
    cd fastmob
    bash scripts/setup_env.sh
    source .venv/bin/activate
    ```

=== "Rebuild Rust extension"

    ```bash
    maturin develop
    ```

=== "Docs"

    ```bash
    uv sync --group docs
    uv run --group docs zensical serve
    uv run --group docs zensical build
    ```

After setup the compiled Rust extension is placed directly in `fastmob/`, so
the package is importable from the repo root without a separate `pip install`.

## Optional extras

| Extra | Installs | Used by |
|---|---|---|
| `geo` | GeoPandas, Shapely, PyProj | geospatial data conversion and tessellation; H3 is native Rust |
| `vis` | `fastmob-vis` | `fastmob.vis` generic ECharts visualization |
