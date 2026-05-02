# Generation Models

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

---

::: skmob2.models.epr.populate_od_matrix
    options:
      show_source: false
