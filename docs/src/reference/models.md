# Generation Models

| API | Description |
| --- | --- |
| [`Gravity`](#skmob2.models.gravity.Gravity) | Gravity model compatible with original scikit-mobility generation APIs. |
| [`Radiation`](#skmob2.models.radiation.Radiation) | Radiation model compatible with original scikit-mobility generation APIs. |
| [`compute_distance_matrix`](#skmob2.models.gravity.compute_distance_matrix) | Compute pairwise Haversine distances for tessellation locations. |
| [`exponential_deterrence_func`](#skmob2.models.gravity.exponential_deterrence_func) | Compute exponential distance deterrence values. |
| [`powerlaw_deterrence_func`](#skmob2.models.gravity.powerlaw_deterrence_func) | Compute power-law distance deterrence values. |
| [`EPR`](#skmob2.models.epr.EPR) | Exploration and Preferential Return trajectory generator. |
| [`DensityEPR`](#skmob2.models.epr.DensityEPR) | EPR generator using location relevance from density. |
| [`SpatialEPR`](#skmob2.models.epr.SpatialEPR) | EPR generator using a spatial origin-destination matrix. |
| [`Ditras`](#skmob2.models.epr.Ditras) | DITRAS trajectory generator combining EPR with mobility diaries. |
| [`MarkovDiaryGenerator`](#skmob2.models.markov_diary_generator.MarkovDiaryGenerator) | Markov Diary Learner and Generator compatible with scikit-mobility. |
| [`GeoSim`](#skmob2.models.geosim.GeoSim) | Social trajectory generator based on GeoSim dynamics. |
| [`STS_epr`](#skmob2.models.sts_epr.STS_epr) | Spatial-temporal social EPR trajectory generator. |
| [`compute_od_matrix`](#skmob2.models.epr.compute_od_matrix) | Compute an OD probability matrix from a singly constrained gravity model. |

The `skmob2.models` package mirrors the original scikit-mobility generation API. Model implementations accept pandas or Narwhals-compatible eager dataframes and may convert inputs to pandas internally.

Install the generation dependencies with:

```bash
pip install "skmob2[generation]"
```

## Flow Models

::: skmob2.models.gravity.Gravity
    options:
      show_source: false

---

::: skmob2.models.radiation.Radiation
    options:
      show_source: false

---

::: skmob2.models.gravity.compute_distance_matrix
    options:
      show_source: false

---

::: skmob2.models.gravity.exponential_deterrence_func
    options:
      show_source: false

---

::: skmob2.models.gravity.powerlaw_deterrence_func
    options:
      show_source: false

## Individual Trajectory Models

::: skmob2.models.epr.EPR
    options:
      show_source: false

---

::: skmob2.models.epr.DensityEPR
    options:
      show_source: false

---

::: skmob2.models.epr.SpatialEPR
    options:
      show_source: false

---

::: skmob2.models.epr.Ditras
    options:
      show_source: false

---

::: skmob2.models.markov_diary_generator.MarkovDiaryGenerator
    options:
      show_source: false

## Social Trajectory Models

::: skmob2.models.geosim.GeoSim
    options:
      show_source: false

---

::: skmob2.models.sts_epr.STS_epr
    options:
      show_source: false

## Helpers

::: skmob2.models.epr.compute_od_matrix
    options:
      show_source: false
