# Data

| API | Description |
| --- | --- |
| [`list_datasets`](#fastmob.data.list_datasets) | List available scikit-mobility datasets. |
| [`load_dataset`](#fastmob.data.load_dataset) | Load one of the original scikit-mobility datasets. |
| [`get_dataset_info`](#fastmob.data.get_dataset_info) | Return metadata for a dataset. |
| [`load_geolife`](#fastmob.data.load_geolife) | Load selected users of the Microsoft GeoLife GPS trajectories, downloading only what is needed. |
| [`load_brightkite`](#fastmob.data.load_brightkite) | Load the first rows of the Brightkite check-in dataset. |

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

---

::: fastmob.data.load_dataset
    options:
      show_source: false

---

::: fastmob.data.get_dataset_info
    options:
      show_source: false

---

::: fastmob.data.load_geolife
    options:
      show_source: false

---

::: fastmob.data.load_brightkite
    options:
      show_source: false
