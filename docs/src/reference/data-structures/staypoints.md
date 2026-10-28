---
icon: lucide/circle-pause
---

# Staypoints

`Staypoints` represents intervals during which a person remained at a place.
It is normally created from `Positionfixes.generate_staypoints()`, and keeps the
metadata needed to resolve its user, coordinates, start time, and end time.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`Staypoints`](#fastmob.core.staypoints_dataframe.Staypoints) | One row per detected stop: a place a user stayed for a while. | [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) |
| [`Staypoints.associate_global_locations`](#fastmob.core.staypoints_dataframe.Staypoints.associate_global_locations) | Validate preassigned exact global IDs against a location catalogue. | [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) |
| [`Staypoints.build_stvd`](#fastmob.core.staypoints_dataframe.Staypoints.build_stvd) | Aggregate these staypoints against a global Locations catalogue into an STVD frame. | [Compare when things happen](../../learn/fit-and-evaluate.md#compare-when-things-happen-spatio-temporal-distance) |
| [`Staypoints.coerce`](#fastmob.core.staypoints_dataframe.Staypoints.coerce) | Return a validated Staypoints wrapper for `value`. | [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs) |
| [`Staypoints.create_activity_flag`](#fastmob.core.staypoints_dataframe.Staypoints.create_activity_flag) | Flag each staypoint as a genuine "activity" by dwell time. | [Which stops are activities?](../../learn/trajectory-hierarchy.md#which-stops-are-activities) |
| [`Staypoints.generate_daily_motifs`](#fastmob.core.staypoints_dataframe.Staypoints.generate_daily_motifs) | Compute one home-anchored mobility motif per user and day. | [One motif per person and day, from the typed tables](../../learn/motifs.md#one-motif-per-person-and-day-from-the-typed-tables) |
| [`Staypoints.generate_global_locations`](#fastmob.core.staypoints_dataframe.Staypoints.generate_global_locations) | Assign staypoints to shared H3 locations and return their catalogue. | [Law 2](../../learn/compare-to-mobility-laws.md#law-2-the-universal-visitation-law) |
| [`Staypoints.generate_user_locations`](#fastmob.core.staypoints_dataframe.Staypoints.generate_user_locations) | Cluster each user's staypoints into recurring locations. | [Build the visits](../../learn/motifs.md#build-the-visits) |
| [`Staypoints.interest_network`](#fastmob.core.staypoints_dataframe.Staypoints.interest_network) | Return the collective interest network over global `locations`. | [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy) |
| [`Staypoints.resolve_dataframe`](#fastmob.core.staypoints_dataframe.Staypoints.resolve_dataframe) | Normalize staypoint input, resolve measure columns, and validate starts. | [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs) |
| [`Staypoints.validate`](#fastmob.core.staypoints_dataframe.Staypoints.validate) | Check required interval columns and chronological ordering. | [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs) |
<!-- api-table:end -->

## Required data

Each row needs a start timestamp and, when validation is enabled, an end
timestamp no earlier than the start. Latitude and longitude are used by location
generation. `staypoint_id` is added during hierarchy generation and connects
staypoints to later trips.

```python
locations, assigned = staypoints.generate_user_locations(epsilon_km=0.1)
active = assigned.create_activity_flag(time_threshold_min=15)
```

Use user-scoped locations for personal recurring places, or
`generate_global_locations()` when multiple users need a shared catalogue.

## API

::: fastmob.core.staypoints_dataframe.Staypoints
    options:
      show_source: false

<!-- example-link -->**Worked example:** [City-wide places instead of per-person places](../../learn/trajectory-hierarchy.md#city-wide-places-instead-of-per-person-places) · [Compare when things happen](../../learn/fit-and-evaluate.md#compare-when-things-happen-spatio-temporal-distance) · [Checking inputs](../../learn/trajectory-hierarchy.md#checking-inputs) · [Which stops are activities?](../../learn/trajectory-hierarchy.md#which-stops-are-activities) · [One motif per person and day, from the typed tables](../../learn/motifs.md#one-motif-per-person-and-day-from-the-typed-tables) · [Law 2](../../learn/compare-to-mobility-laws.md#law-2-the-universal-visitation-law) · [Build the visits](../../learn/motifs.md#build-the-visits) · [When and where is each place busy?](../../learn/population-measures.md#when-and-where-is-each-place-busy)

