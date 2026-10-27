---
icon: lucide/route
---

# Triplegs

`Triplegs` summarizes one movement segment between consecutive staypoints. Each
row describes a door-to-door segment rather than its raw point geometry, making
it suitable for mode classification and trip aggregation.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Triplegs`](#fastmob.core.triplegs_dataframe.Triplegs) | One row per tripleg (a movement segment between two staypoints). |  |
| [`calculate_modal_split`](#fastmob.core.triplegs_dataframe.Triplegs.calculate_modal_split) | Aggregate this `mode`-labeled table into a modal-split table. | [Movement between stops](../../learn/trajectory-hierarchy.md#movement-between-stops-triplegs-and-transport-mode) |
| [`from_positionfixes`](#fastmob.core.triplegs_dataframe.Triplegs.from_positionfixes) | Derive triplegs from positionfixes + already-generated staypoints. | [The same pipeline, step by step](../../learn/trajectory-hierarchy.md#the-same-pipeline-step-by-step) |
| [`generate_trips`](#fastmob.core.triplegs_dataframe.Triplegs.generate_trips) | Group consecutive triplegs into trips. | [Trips and tours](../../learn/trajectory-hierarchy.md#trips-and-tours) |
| [`predict_transport_mode`](#fastmob.core.triplegs_dataframe.Triplegs.predict_transport_mode) | Classify each tripleg's transport mode from its average speed. | [Movement between stops](../../learn/trajectory-hierarchy.md#movement-between-stops-triplegs-and-transport-mode) |
<!-- api-table:end -->

## Required data

Validated triplegs contain `tripleg_id`, `started_at`, `finished_at`,
`length_km`, and `duration_s`, plus a user ID when the data represents multiple
people. Generated triplegs also include mean speed information.

```python
classified = triplegs.predict_transport_mode()
trips = classified.generate_trips(active_staypoints)
```

`generate_trips()` requires staypoints with an `activity` flag. Create that flag
first with `Staypoints.create_activity_flag()`.

## API

::: fastmob.core.triplegs_dataframe.Triplegs
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Movement between stops](../../learn/trajectory-hierarchy.md#movement-between-stops-triplegs-and-transport-mode) · [The same pipeline, step by step](../../learn/trajectory-hierarchy.md#the-same-pipeline-step-by-step) · [Trips and tours](../../learn/trajectory-hierarchy.md#trips-and-tours)

