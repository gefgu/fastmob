# Generation Models

| API | Description |
| --- | --- |
| [`Gravity`](#fastmob.models.gravity.Gravity) | Gravity model compatible with original scikit-mobility generation APIs. |
| [`Radiation`](#fastmob.models.radiation.Radiation) | Radiation model compatible with original scikit-mobility generation APIs. |
| [`EPR`](#fastmob.models.epr.EPR) | Exploration and Preferential Return trajectory generator. |
| [`DensityEPR`](#fastmob.models.epr.DensityEPR) | EPR generator using location relevance from density. |
| [`SpatialEPR`](#fastmob.models.epr.SpatialEPR) | EPR generator using a spatial origin-destination matrix. |
| [`Ditras`](#fastmob.models.epr.Ditras) | DITRAS trajectory generator combining EPR with mobility diaries. |
| [`MarkovDiaryGenerator`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator) | Markov Diary Learner and Generator compatible with scikit-mobility. |
| [`GeoSim`](#fastmob.models.geosim.GeoSim) | Social trajectory generator based on GeoSim dynamics. |
| [`STS_epr`](#fastmob.models.sts_epr.STS_epr) | Spatial-temporal social EPR trajectory generator. |

The `fastmob.models` package mirrors the original scikit-mobility generation API. Model implementations accept pandas or Narwhals-compatible eager dataframes and may convert inputs to pandas internally.

Only `Gravity.fit()` needs Statsmodels for Poisson-GLM fitting:

```bash
pip install statsmodels
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
