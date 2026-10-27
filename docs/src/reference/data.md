# Data

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`list_datasets`](#fastmob.data.list_datasets) | List datasets. | [Which datasets does the library list?](../learn/process-trajectories.md#which-datasets-does-the-library-list) |
| [`load_dataset`](#fastmob.data.load_dataset) | Load dataset. |  |
| [`get_dataset_info`](#fastmob.data.get_dataset_info) | Get dataset info. | [Which datasets does the library list?](../learn/process-trajectories.md#which-datasets-does-the-library-list) |
| [`load_geolife`](#fastmob.data.load_geolife) | Load GPS trajectories of selected GeoLife users, downloading only what is needed. | [Load the data](../learn/clean-and-detect-stops.md#load-the-data) |
| [`load_brightkite`](#fastmob.data.load_brightkite) | Load Brightkite location check-ins, downloading only the rows requested. | [Load and clean](../learn/compare-to-mobility-laws.md#load-and-clean) |
| [`DatasetBuilder`](#fastmob.data.DatasetBuilder) | Base class for dataset-specific builders. |  |
| [`prepare`](#fastmob.data.DatasetBuilder.prepare) | Turn the downloaded files (a list of paths) into the dataset object; implemented by each dataset builder. |  |
<!-- api-table:end -->

## Datasets

| Name | Type | Description |
| ---- | ---- | ----------- |
| `foursquare_nyc` | trajectory | Foursquare checkins of individuals in New York City. |
| `flow_foursquare_nyc` | flow | Movements between 4 km tiles in NYC derived from Foursquare data. |
| `taxi_san_francisco` | trajectory | GPS traces of ~500 taxis over 30 days in the San Francisco Bay Area. Requires auth. |
| `parking_san_francisco` | auxiliar | Parking meters in San Francisco. |
| `nyc_boundaries` | shape | NYC Borough Boundaries shapefile (water areas included). |

## API Reference

::: fastmob.data.list_datasets
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Which datasets does the library list?](../learn/process-trajectories.md#which-datasets-does-the-library-list)

---

::: fastmob.data.load_dataset
    options:
      show_source: false

---

::: fastmob.data.get_dataset_info
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Which datasets does the library list?](../learn/process-trajectories.md#which-datasets-does-the-library-list)

---

::: fastmob.data.load_geolife
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Load the data](../learn/clean-and-detect-stops.md#load-the-data)

---

::: fastmob.data.load_brightkite
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Load and clean](../learn/compare-to-mobility-laws.md#load-and-clean)

---

::: fastmob.data.DatasetBuilder
    options:
      show_source: false
