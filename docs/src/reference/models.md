# Generation Models

<!-- api-table:start -->
**Flow models**

| API | Description | Example |
| --- | --- | --- |
| [`Gravity`](#fastmob.models.gravity.Gravity) | Gravity model: flows between places from their sizes and the distance between them. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Gravity.fit`](#fastmob.models.gravity.Gravity.fit) | Fit the Gravity model parameters to observed flows via Poisson regression. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Gravity.generate`](#fastmob.models.gravity.Gravity.generate) | Generate synthetic flows with the Gravity model. | [Fit and run a gravity model](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`Radiation`](#fastmob.models.radiation.Radiation) | Radiation model: flows between places from their populations and the opportunities in between. | [Flows](../learn/generate-mobility.md#flows-gravity-and-radiation-models) |
| [`Radiation.generate`](#fastmob.models.radiation.Radiation.generate) | Generate synthetic flows with the Radiation model. | [Flows](../learn/generate-mobility.md#flows-gravity-and-radiation-models) |

**Individual trajectory models**

| API | Description | Example |
| --- | --- | --- |
| [`EPR`](#fastmob.models.epr.EPR) | Exploration and Preferential Return (EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`EPR.generate`](#fastmob.models.epr.EPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`DensityEPR`](#fastmob.models.epr.DensityEPR) | Density-EPR (d-EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`DensityEPR.generate`](#fastmob.models.epr.DensityEPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`SpatialEPR`](#fastmob.models.epr.SpatialEPR) | Spatial-EPR (s-EPR) trajectory generator. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`SpatialEPR.generate`](#fastmob.models.epr.SpatialEPR.generate) | Simulate agents from `start_date` to `end_date`. | [Exploration and preferential return](../learn/generate-mobility.md#exploration-and-preferential-return) |
| [`Ditras`](#fastmob.models.epr.Ditras) | DITRAS (DIary-based TRAjectory Simulator) modelling framework. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`Ditras.generate`](#fastmob.models.epr.Ditras.generate) | Simulate agents from `start_date` to `end_date`. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator) | Markov diary learner and generator: when a person is at home or away, hour by hour. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator.fit`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator.fit) | Learn the Markov mobility diary from real trajectories. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |
| [`MarkovDiaryGenerator.generate`](#fastmob.models.markov_diary_generator.MarkovDiaryGenerator.generate) | Generate a synthetic mobility diary. | [Realistic daily rhythm](../learn/generate-mobility.md#realistic-daily-rhythm-diaries) |

**Social trajectory models**

| API | Description | Example |
| --- | --- | --- |
| [`GeoSim`](#fastmob.models.geosim.GeoSim) | GeoSim social trajectory generator. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`GeoSim.generate`](#fastmob.models.geosim.GeoSim.generate) | Simulate a socially connected population from `start_date` to `end_date`. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`STS_epr`](#fastmob.models.sts_epr.STS_epr) | STS-EPR (Spatial, Temporal, and Social EPR) trajectory generator. | [Social ties](../learn/generate-mobility.md#social-ties) |
| [`STS_epr.generate`](#fastmob.models.sts_epr.STS_epr.generate) | Simulate a socially connected population from `start_date` to `end_date`. | [Social ties](../learn/generate-mobility.md#social-ties) |

**Next-location prediction**

| API | Description | Example |
| --- | --- | --- |
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

### Fitting flows with custom column names

`Gravity.fit` accepts a `FlowDataFrame` with an attached `Locations` catalogue,
or a dataframe or `FlowDataFrame` with the catalogue passed as `locations=`.
The origin and destination values must match its location IDs, and the catalogue
must contain the selected relevance column (population or another measure of
location size). For an existing pandas table `flows_df` with columns `Area_O`,
`Area_D`, and `Weight_Day`, and a prepared `locations` catalogue:

```python
from fastmob.core import FlowDataFrame
from fastmob.models import Gravity

flows = FlowDataFrame(flows_df, locations=locations)
model = Gravity()
model.fit(
    flows,
    origin_col="Area_O",
    destination_col="Area_D",
    flow_col="Weight_Day",
    relevance_column="relevance",
)
```

For an OD table without an attached catalogue, pass it directly:

```python
model.fit(flows_df, locations=locations, origin_col="Area_O",
          destination_col="Area_D", flow_col="Weight_Day")
```

An explicit `locations` argument takes precedence over an attached catalogue.

The default names are `origin`, `destination`, and `flow`. Alternatively,
`FlowDataFrame` can rename input columns to those defaults using its `origin`,
`destination`, and `flow` constructor arguments; in that case use the default
names when fitting. See the [worked example](../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model)
for preparing locations and interpreting a fit.

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

## Next-location prediction

::: fastmob.models.next_location.NextLocationPredictor
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Predict the next place](../learn/visit-based-measures.md#predict-the-next-place)

