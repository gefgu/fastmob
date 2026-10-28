# Visualization

Install the optional visualization package with `pip install "fastmob[vis]"`. Its stable
entry point is `fastmob.vis`; the wheel is named `fastmob-vis` and the module behind it is
`fastmob_vis`. The generic chart constructors are exported by `fastmob.vis`. The charts for
particular results (jump lengths, activities, mobility profiles, motifs, fitted laws and
STVD) are in `fastmob_vis`.

Every constructor takes a pandas, Polars or other eager dataframe and the names of the
columns to draw. Generic constructors return an immutable `Chart`; the result-specific
functions return an `EChartsFigure`. Both draw themselves as the last expression of a
notebook cell, and export to a web page, to JSON for a web application, or to a static SVG.
For a walk-through that draws every chart, see
[Chart your results](../learn/visualize-results.md).

<!-- api-table:start -->
**Generic charts from a table**

| API | Description | Example |
| --- | --- | --- |
| [`ecdf`](#fastmob_vis.charts.ecdf) | Draw the empirical cumulative distribution of a numeric column, optionally against a second result. | [A distribution from one result](../learn/visualize-results.md#a-distribution-from-one-result) |
| [`bar`](#fastmob_vis.charts.bar) | Draw a categorical bar chart. | [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table) |
| [`scatter`](#fastmob_vis.charts.scatter) | Draw a scatter chart of two numeric columns. | [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table) |
| [`heatmap`](#fastmob_vis.charts.heatmap) | Draw a heatmap from a long-form table of x, y and value columns. | [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table) |
| [`boxplot`](#fastmob_vis.charts.boxplot) | Draw a box plot of a numeric column for each category. | [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table) |
| [`map`](#fastmob_vis.charts.map) | Draw polygons from a GeoJSON FeatureCollection as a map chart. | [Maps, which are interactive only](../learn/visualize-results.md#maps-which-are-interactive-only) |

**Chart objects and export**

| API | Description | Example |
| --- | --- | --- |
| [`Chart`](#fastmob_vis.charts.Chart) | Immutable, notebook-renderable chart specification. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`Chart.render`](#fastmob_vis.charts.Chart.render) | Return the lower-level `EChartsFigure` with the same option and settings. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`Chart.to_dict`](#fastmob_vis.charts.Chart.to_dict) | Return a copy of the ECharts option, ready to send to a web page. |  |
| [`Chart.to_html`](#fastmob_vis.charts.Chart.to_html) | Write a standalone HTML page that draws the chart. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`Chart.to_json`](#fastmob_vis.charts.Chart.to_json) | Write the ECharts option as a JSON file. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`Chart.to_svg`](#fastmob_vis.charts.Chart.to_svg) | Draw the chart as a static SVG picture. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`EChartsFigure`](#fastmob_vis.figure.EChartsFigure) | Low-level notebook-renderable ECharts option. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`EChartsFigure.to_dict`](#fastmob_vis.figure.EChartsFigure.to_dict) | Return a copy of the ECharts option, ready to send to a web page. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |
| [`EChartsFigure.to_html`](#fastmob_vis.figure.EChartsFigure.to_html) | Write a standalone HTML page that draws the chart. |  |
| [`EChartsFigure.to_json`](#fastmob_vis.figure.EChartsFigure.to_json) | Write the ECharts option as a JSON file. |  |
| [`EChartsFigure.to_svg`](#fastmob_vis.figure.EChartsFigure.to_svg) | Draw the chart as a static SVG picture. |  |
| [`get_resource_bundle`](#fastmob_vis.figure.get_resource_bundle) | Return the script and style tags that charts need, to include once in a page. | [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts) |

**Distributions of results (ECDF)**

| API | Description | Example |
| --- | --- | --- |
| [`plot_jump_lengths_ecdf`](#fastmob_vis.ecdf.plot_jump_lengths_ecdf) | Compare two sets of jump lengths with ECDF curves. | [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result) |
| [`plot_radius_of_gyration_ecdf`](#fastmob_vis.ecdf.plot_radius_of_gyration_ecdf) | Compare two sets of radii of gyration with ECDF curves. | [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result) |
| [`plot_visits_frequency_ecdf`](#fastmob_vis.ecdf.plot_visits_frequency_ecdf) | Compare two sets of visit counts with ECDF curves. | [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result) |
| [`plot_trip_duration_ecdf`](#fastmob_vis.ecdf.plot_trip_duration_ecdf) | Compare two sets of trip durations with ECDF curves. | [Time spent and time between visits](../learn/visualize-results.md#time-spent-and-time-between-visits) |
| [`plot_dwell_time_ecdf`](#fastmob_vis.ecdf.plot_dwell_time_ecdf) | Compare two sets of dwell times with ECDF curves. | [Time spent and time between visits](../learn/visualize-results.md#time-spent-and-time-between-visits) |

**Visits and activities**

| API | Description | Example |
| --- | --- | --- |
| [`plot_visit_purpose_distribution`](#fastmob_vis.activity.plot_visit_purpose_distribution) | Draw the share of visits by purpose as a bar chart. | [Visit purposes](../learn/visualize-results.md#visit-purposes) |
| [`plot_visit_purpose_comparison`](#fastmob_vis.activity.plot_visit_purpose_comparison) | Compare visit-purpose shares of several datasets as grouped bars. | [Visit purposes](../learn/visualize-results.md#visit-purposes) |
| [`plot_activity_transition_matrix`](#fastmob_vis.activity.plot_activity_transition_matrix) | Draw the activity-to-activity transition matrix as a heatmap. | [Moving between activities](../learn/visualize-results.md#moving-between-activities) |
| [`plot_activity_transition_difference`](#fastmob_vis.activity.plot_activity_transition_difference) | Draw the difference between two transition matrices as a signed heatmap. | [Moving between activities](../learn/visualize-results.md#moving-between-activities) |
| [`plot_daily_activity_distribution`](#fastmob_vis.activity.plot_daily_activity_distribution) | Draw the daily activity distribution (DARD): the share of each activity by time of day. | [Activities through the day](../learn/visualize-results.md#activities-through-the-day) |
| [`plot_daily_activity_difference`](#fastmob_vis.activity.plot_daily_activity_difference) | Draw the difference between two daily activity distributions as a signed heatmap. | [Activities through the day](../learn/visualize-results.md#activities-through-the-day) |

**Mobility profiles and motifs**

| API | Description | Example |
| --- | --- | --- |
| [`plot_mobility_profiles`](#fastmob_vis.profiles.plot_mobility_profiles) | Scatter intermittency against degree of return, coloured by mobility profile. | [Mobility profiles](../learn/visualize-results.md#mobility-profiles) |
| [`plot_profile_metrics`](#fastmob_vis.profiles.plot_profile_metrics) | Draw box plots of mobility metrics for each mobility profile, one panel per metric. | [Mobility profiles](../learn/visualize-results.md#mobility-profiles) |
| [`plot_motif_literature_comparison`](#fastmob_vis.motifs.plot_motif_literature_comparison) | Compare motif distributions with the literature as grouped bars. | [Daily motifs](../learn/visualize-results.md#daily-motifs) |

**Fitted mobility laws**

| API | Description | Example |
| --- | --- | --- |
| [`plot_truncated_powerlaw_fits`](#fastmob_vis.mobility_laws.plot_truncated_powerlaw_fits) | Plot jump-length distributions with fitted truncated power laws. | [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits) |
| [`plot_lognormal_fits`](#fastmob_vis.mobility_laws.plot_lognormal_fits) | Plot the distribution of daily visited locations with fitted log-normal curves. | [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits) |
| [`plot_distance_frequency_law`](#fastmob_vis.mobility_laws.plot_distance_frequency_law) | Plot the distance-frequency visitation law with fitted curves. | [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits) |

**Maps**

| API | Description | Example |
| --- | --- | --- |
| [`plot_stvd_comparison`](#fastmob_vis.stvd.plot_stvd_comparison) | Map a spatio-temporal visitation difference (STVD) comparison on a Leaflet basemap. | [Maps, which are interactive only](../learn/visualize-results.md#maps-which-are-interactive-only) |
<!-- api-table:end -->

## Choosing the output

Pass `display="svg"` to any constructor to show a static picture in notebooks that do not
run JavaScript (documentation, some viewers). The default `display="html"` is interactive.
Maps (`map`, `plot_stvd_comparison`) are interactive only. To put several charts on one
page, write the libraries once with `get_resource_bundle` and make the charts with
`bundle_libs=False`.

## Palettes

`palette` takes the name of one of four built-in palettes, or a mapping with the same keys
(`bg`, `grid`, `axis`, `observed`, `synthetic`, `syn2`, `syn3`, `baseline`, `band`). The
first series is drawn in `observed`, the second in `synthetic`, and further series in
`syn2`, `syn3` and `baseline`. `fastmob.vis.PALETTES` holds them.

| Palette | Background | Observed | Synthetic | Third | Fourth | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `warm` (default) | `#fbf8f1` | `#14110d` | `#c8533a` | `#7a8c5a` | `#3d6a8a` | `#a39c8e` |
| `ink` | `#fbf8f1` | `#14110d` | `#5a544a` | `#7a7468` | `#403b34` | `#bfb8a8` |
| `terracotta` | `#fbf8f1` | `#3a1e15` | `#c8533a` | `#e08a4a` | `#7a3325` | `#b89a8e` |
| `forest` | `#f4efe3` | `#1e2a1a` | `#4e7a3d` | `#8a9a4a` | `#2d4a3e` | `#a39c8e` |

## Motif reference data

`fastmob.vis.LITERATURE_MOTIF_PERCENTAGES` maps the ordinal (1 to 17) of each daily motif of
the literature to its percentage of person-days, and `fastmob.vis.LITERATURE_TO_FASTMOB_MOTIF_ID`
maps the same ordinal to the packed integer id that `motif_distribution` uses. They are what
`plot_motif_literature_comparison` compares against.

```python
from fastmob.vis import LITERATURE_MOTIF_PERCENTAGES, LITERATURE_TO_FASTMOB_MOTIF_ID
```

The former Folium and Matplotlib plotting API was removed when `fastmob-vis` became a
separate package. Use the charts below or a dedicated GIS renderer for maps with basemaps.

---

::: fastmob_vis.charts.ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [A distribution from one result](../learn/visualize-results.md#a-distribution-from-one-result)

---

::: fastmob_vis.charts.bar
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table)

---

::: fastmob_vis.charts.scatter
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table)

---

::: fastmob_vis.charts.heatmap
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table)

---

::: fastmob_vis.charts.boxplot
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Bars, boxes, points and a heatmap from a table](../learn/visualize-results.md#bars-boxes-points-and-a-heatmap-from-a-table)

---

::: fastmob_vis.charts.map
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Maps, which are interactive only](../learn/visualize-results.md#maps-which-are-interactive-only)

---

::: fastmob_vis.charts.Chart
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts)

---

::: fastmob_vis.figure.EChartsFigure
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts)

---

::: fastmob_vis.figure.get_resource_bundle
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Save and embed charts](../learn/visualize-results.md#save-and-embed-charts)

---

::: fastmob_vis.ecdf.plot_jump_lengths_ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result)

---

::: fastmob_vis.ecdf.plot_radius_of_gyration_ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result)

---

::: fastmob_vis.ecdf.plot_visits_frequency_ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Charts for one kind of result](../learn/visualize-results.md#charts-for-one-kind-of-result)

---

::: fastmob_vis.ecdf.plot_trip_duration_ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Time spent and time between visits](../learn/visualize-results.md#time-spent-and-time-between-visits)

---

::: fastmob_vis.ecdf.plot_dwell_time_ecdf
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Time spent and time between visits](../learn/visualize-results.md#time-spent-and-time-between-visits)

---

::: fastmob_vis.activity.plot_visit_purpose_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Visit purposes](../learn/visualize-results.md#visit-purposes)

---

::: fastmob_vis.activity.plot_visit_purpose_comparison
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Visit purposes](../learn/visualize-results.md#visit-purposes)

---

::: fastmob_vis.activity.plot_activity_transition_matrix
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Moving between activities](../learn/visualize-results.md#moving-between-activities)

---

::: fastmob_vis.activity.plot_activity_transition_difference
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Moving between activities](../learn/visualize-results.md#moving-between-activities)

---

::: fastmob_vis.activity.plot_daily_activity_distribution
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Activities through the day](../learn/visualize-results.md#activities-through-the-day)

---

::: fastmob_vis.activity.plot_daily_activity_difference
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Activities through the day](../learn/visualize-results.md#activities-through-the-day)

---

::: fastmob_vis.profiles.plot_mobility_profiles
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Mobility profiles](../learn/visualize-results.md#mobility-profiles)

---

::: fastmob_vis.profiles.plot_profile_metrics
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Mobility profiles](../learn/visualize-results.md#mobility-profiles)

---

::: fastmob_vis.motifs.plot_motif_literature_comparison
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Daily motifs](../learn/visualize-results.md#daily-motifs)

---

::: fastmob_vis.mobility_laws.plot_truncated_powerlaw_fits
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits)

---

::: fastmob_vis.mobility_laws.plot_lognormal_fits
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits)

---

::: fastmob_vis.mobility_laws.plot_distance_frequency_law
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Mobility laws and their fits](../learn/visualize-results.md#mobility-laws-and-their-fits)

---

::: fastmob_vis.stvd.plot_stvd_comparison
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Maps, which are interactive only](../learn/visualize-results.md#maps-which-are-interactive-only)

