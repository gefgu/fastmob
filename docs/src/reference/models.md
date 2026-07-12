# Generation Models

| API | Description |
| --- | --- |
| [`Gravity`](#fastmob.models.gravity.Gravity) | Gravity model compatible with original scikit-mobility generation APIs. |
| [`Radiation`](#fastmob.models.radiation.Radiation) | Radiation model compatible with original scikit-mobility generation APIs. |
| [`compute_distance_matrix`](#fastmob.models.gravity.compute_distance_matrix) | Compute pairwise Haversine distances for tessellation locations. |
| [`exponential_deterrence_func`](#fastmob.models.gravity.exponential_deterrence_func) | Compute exponential distance deterrence values. |
| [`powerlaw_deterrence_func`](#fastmob.models.gravity.powerlaw_deterrence_func) | Compute power-law distance deterrence values. |
| [`EPR`](#fastmob.models.epr.EPR) | Exploration and Preferential Return trajectory generator. |
| [`DensityEPR`](#fastmob.models.epr.DensityEPR) | EPR generator using location relevance from density. |
| [`SpatialEPR`](#fastmob.models.epr.SpatialEPR) | EPR generator using a spatial origin-destination matrix. |
| [`Ditras`](#fastmob.models.epr.Ditras) | DITRAS trajectory generator combining EPR with mobility diaries. |
| [`MarkovDiaryGenerator`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator) | Markov Diary Learner and Generator compatible with scikit-mobility. |
| [`GeoSim`](#fastmob.models.geosim.GeoSim) | Social trajectory generator based on GeoSim dynamics. |
| [`STS_epr`](#fastmob.models.sts_epr.STS_epr) | Spatial-temporal social EPR trajectory generator. |
| [`compute_od_matrix`](#fastmob.models.epr.compute_od_matrix) | Compute an OD probability matrix from a singly constrained gravity model. |

The `fastmob.models` package mirrors the original scikit-mobility generation API. Model implementations accept pandas or Narwhals-compatible eager dataframes and may convert inputs to pandas internally.

Install the generation dependencies with:

```bash
pip install "fastmob[generation]"
```

## Flow Models

::: fastmob.models.gravity.Gravity
    options:
      show_source: false

---

::: fastmob.models.radiation.Radiation
    options:
      show_source: false

---

::: fastmob.models.gravity.compute_distance_matrix
    options:
      show_source: false

---

::: fastmob.models.gravity.exponential_deterrence_func
    options:
      show_source: false

---

::: fastmob.models.gravity.powerlaw_deterrence_func
    options:
      show_source: false

## Individual Trajectory Models

::: fastmob.models.epr.EPR
    options:
      show_source: false

---

::: fastmob.models.epr.DensityEPR
    options:
      show_source: false

---

::: fastmob.models.epr.SpatialEPR
    options:
      show_source: false

---

::: fastmob.models.epr.Ditras
    options:
      show_source: false

---

::: fastmob.models.markov_diary_generator.MarkovDiaryGenerator
    options:
      show_source: false

## Social Trajectory Models

::: fastmob.models.geosim.GeoSim
    options:
      show_source: false

---

::: fastmob.models.sts_epr.STS_epr
    options:
      show_source: false

## Helpers

::: fastmob.models.epr.compute_od_matrix
    options:
      show_source: false
