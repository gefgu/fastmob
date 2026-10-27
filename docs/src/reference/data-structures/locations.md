---
icon: lucide/map-pinned
---

# Locations

`Locations` is a catalogue of meaningful places. A catalogue is either
user-scoped, where a location ID is meaningful only for one user, or global,
where every user shares the same location IDs. This distinction makes OD and
collective analysis explicit.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Locations`](#fastmob.core.locations_dataframe.Locations) | One row per location in either a user or global scope. |  |
| [`from_tessellation`](#fastmob.core.locations_dataframe.Locations.from_tessellation) | Convert a legacy tessellation into a global location catalogue. | [Fit and run a gravity model](../../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) |
| [`identify`](#fastmob.core.locations_dataframe.Locations.identify) | Label each location as `"home"`, `"work"`, or `"other"`. | [Build the visits](../../learn/motifs.md#build-the-visits) |
| [`model_input`](#fastmob.core.locations_dataframe.Locations.model_input) | Return the validated Arrow-ready input used by spatial models. | [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) |
| [`require_global`](#fastmob.core.locations_dataframe.Locations.require_global) | Return validated global locations as a Narwhals eager dataframe. | [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) |
| [`validate_staypoint_assignments`](#fastmob.core.locations_dataframe.Locations.validate_staypoint_assignments) | Validate non-null staypoint location identities against this catalogue. | [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs) |
<!-- api-table:end -->

## Required data

Rows require `location_id`, `center_lat`, and `center_lng`. User-scoped catalogues
also carry a user-ID column; global catalogues must not. The `scope` and `scheme`
metadata describe identity and whether IDs came from clustering, H3, or an
external source.

```python
locations, assigned = staypoints.generate_user_locations(epsilon_km=0.1)
labelled = locations.identify(assigned)
```

Global catalogues validate assignments used by models and collective measures;
user catalogues can be labeled as home, work, or other.

## API

::: fastmob.core.locations_dataframe.Locations
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Fit and run a gravity model](../../learn/fit-and-evaluate.md#fit-and-run-a-gravity-model) · [Build the visits](../../learn/motifs.md#build-the-visits) · [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) · [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs)

