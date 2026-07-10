# Generation Models

| API | Description |
| --- | --- |
| [`Gravity`](#fkmob.models.gravity.Gravity) | Gravity model compatible with original scikit-mobility generation APIs. |
| [`Radiation`](#fkmob.models.radiation.Radiation) | Radiation model compatible with original scikit-mobility generation APIs. |
| [`compute_distance_matrix`](#fkmob.models.gravity.compute_distance_matrix) | Compute pairwise Haversine distances for tessellation locations. |
| [`exponential_deterrence_func`](#fkmob.models.gravity.exponential_deterrence_func) | Compute exponential distance deterrence values. |
| [`powerlaw_deterrence_func`](#fkmob.models.gravity.powerlaw_deterrence_func) | Compute power-law distance deterrence values. |
| [`EPR`](#fkmob.models.epr.EPR) | Exploration and Preferential Return trajectory generator. |
| [`DensityEPR`](#fkmob.models.epr.DensityEPR) | EPR generator using location relevance from density. |
| [`SpatialEPR`](#fkmob.models.epr.SpatialEPR) | EPR generator using a spatial origin-destination matrix. |
| [`Ditras`](#fkmob.models.epr.Ditras) | DITRAS trajectory generator combining EPR with mobility diaries. |
| [`MarkovDiaryGenerator`](#fkmob.models.markov_diary_generator.MarkovDiaryGenerator) | Markov Diary Learner and Generator compatible with scikit-mobility. |
| [`GeoSim`](#fkmob.models.geosim.GeoSim) | Social trajectory generator based on GeoSim dynamics. |
| [`STS_epr`](#fkmob.models.sts_epr.STS_epr) | Spatial-temporal social EPR trajectory generator. |
| [`compute_od_matrix`](#fkmob.models.epr.compute_od_matrix) | Compute an OD probability matrix from a singly constrained gravity model. |

The `fkmob.models` package mirrors the original scikit-mobility generation API. Model implementations accept pandas or Narwhals-compatible eager dataframes and may convert inputs to pandas internally.

Install the generation dependencies with:

```bash
pip install "fkmob[generation]"
```

## Flow Models

::: fkmob.models.gravity.Gravity
    options:
      show_source: false

---

::: fkmob.models.radiation.Radiation
    options:
      show_source: false

---

::: fkmob.models.gravity.compute_distance_matrix
    options:
      show_source: false

---

::: fkmob.models.gravity.exponential_deterrence_func
    options:
      show_source: false

---

::: fkmob.models.gravity.powerlaw_deterrence_func
    options:
      show_source: false

## Individual Trajectory Models

::: fkmob.models.epr.EPR
    options:
      show_source: false

---

::: fkmob.models.epr.DensityEPR
    options:
      show_source: false

---

::: fkmob.models.epr.SpatialEPR
    options:
      show_source: false

---

::: fkmob.models.epr.Ditras
    options:
      show_source: false

---

::: fkmob.models.markov_diary_generator.MarkovDiaryGenerator
    options:
      show_source: false

## Social Trajectory Models

::: fkmob.models.geosim.GeoSim
    options:
      show_source: false

---

::: fkmob.models.sts_epr.STS_epr
    options:
      show_source: false

## Helpers

::: fkmob.models.epr.compute_od_matrix
    options:
      show_source: false
