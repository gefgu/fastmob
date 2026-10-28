# Generation Models

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Gravity`](#fastmob.models.gravity.Gravity) | Gravity model. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Gravity.fit`](#fastmob.models.gravity.Gravity.fit) | Fit the Gravity model parameters to observed flows via Poisson regression. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Gravity.generate`](#fastmob.models.gravity.Gravity.generate) | Generate synthetic flows with the Gravity model. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Radiation`](#fastmob.models.radiation.Radiation) | Radiation model. | [Flows](../learn/generate-mobility.md#flows-gravity-and-radiation-models) |
| [`Radiation.generate`](#fastmob.models.radiation.Radiation.generate) | Generate synthetic flows with the Radiation model. | [Flows](../learn/generate-mobility.md#flows-gravity-and-radiation-models) |
| [`EPR`](#fastmob.models.epr.EPR) | Exploration and Preferential Return (EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`EPR.generate`](#fastmob.models.epr.EPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`DensityEPR`](#fastmob.models.epr.DensityEPR) | Density-EPR (d-EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`DensityEPR.generate`](#fastmob.models.epr.DensityEPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`SpatialEPR`](#fastmob.models.epr.SpatialEPR) | Spatial-EPR (s-EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`SpatialEPR.generate`](#fastmob.models.epr.SpatialEPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`Ditras`](#fastmob.models.epr.Ditras) | DITRAS (DIary-based TRAjectory Simulator) modelling framework. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`Ditras.generate`](#fastmob.models.epr.Ditras.generate) | Simulate agents from `start_date` to `end_date`. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator) | Markov Diary Learner and Generator. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator.fit`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator.fit) | Learn the Markov mobility diary from real trajectories. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator.generate`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator.generate) | Generate a synthetic mobility diary. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`GeoSim`](#fastmob.models.geosim.GeoSim) | GeoSim social trajectory generator. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`GeoSim.generate`](#fastmob.models.geosim.GeoSim.generate) | Simulate a socially connected population from `start_date` to `end_date`. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`STS_epr`](#fastmob.models.sts_epr.STS_epr) | STS-EPR (Spatial, Temporal, and Social EPR) trajectory generator. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`STS_epr.generate`](#fastmob.models.sts_epr.STS_epr.generate) | Simulate a socially connected population from `start_date` to `end_date`. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`NextLocationPredictor`](#fastmob.models.next_location.NextLocationPredictor) | Order-k Markov next-location predictor with optional backoff. | [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place) |
| [`NextLocationPredictor.evaluate`](#fastmob.models.next_location.NextLocationPredictor.evaluate) | Holdout-last-`n_holdout` accuracy over the data given to `fit`. | [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place) |
| [`NextLocationPredictor.fit`](#fastmob.models.next_location.NextLocationPredictor.fit) | Fit one Markov model per user from each user's location-id sequence. | [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place) |
| [`NextLocationPredictor.predict`](#fastmob.models.next_location.NextLocationPredictor.predict) | Predict each fitted user's single most likely next location. | [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place) |
| [`NextLocationPredictor.predict_proba`](#fastmob.models.next_location.NextLocationPredictor.predict_proba) | Predict each fitted user's next location(s) from their own training history's tail. | [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place) |
<!-- api-table:end -->

The `fastmob.models` package mirrors the original scikit-mobility generation API. Model implementations accept pandas or Narwhals-compatible eager dataframes and may convert inputs to pandas internally.

Only `Gravity.fit()` needs Statsmodels for Poisson-GLM fitting:

```bash
pip install statsmodels
```

## Flow Models

::: fastmob.models.gravity.Gravity
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model)

---

::: fastmob.models.radiation.Radiation
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Flows](../learn/generate-mobility.md#flows-gravity-and-radiation-models)

---

## Individual Trajectory Models

::: fastmob.models.epr.EPR
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return)

---

::: fastmob.models.epr.DensityEPR
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return)

---

::: fastmob.models.epr.SpatialEPR
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return)

---

::: fastmob.models.epr.Ditras
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries)

---

::: fastmob.models.markov_diary_generator.MarkovDiaryGenerator
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries)

## Social Trajectory Models

::: fastmob.models.geosim.GeoSim
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Social ties](../learn/generate-mobility.md#social-ties)

---

::: fastmob.models.sts_epr.STS_epr
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Social ties](../learn/generate-mobility.md#social-ties)

---

::: fastmob.models.next_location.NextLocationPredictor
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place)

