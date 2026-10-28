# BaseDataFrame

The common base of every typed dataframe (`TrajDataFrame`, `Positionfixes`, `Staypoints`, `Locations`,
`Triplegs`, `Trips`, `Tours`, `FlowDataFrame`). It gives them conversion to the native dataframe, a
metric comparison with another table and the chart builders of the optional `fastmob[vis]` extra.

## Catalogue

<!-- api-table:start -->
| API | Description | Example |
| --- | --- | --- |
| [`BaseDataFrame`](#fastmob.core.base.BaseDataFrame) | Small wrapper around an eager dataframe backend. |  |
| [`compare_to`](#fastmob.core.base.BaseDataFrame.compare_to) | Compare `value_col` against `other`, optionally grouped by `group_col`. | [Convert and compare typed tables](../../learn/process-trajectories.md#convert-and-compare-typed-tables) |
| [`plot_bar`](#fastmob.core.base.BaseDataFrame.plot_bar) | Build a categorical bar chart from existing columns. | [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) |
| [`plot_boxplot`](#fastmob.core.base.BaseDataFrame.plot_boxplot) | Build a box plot from existing long-form columns. | [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) |
| [`plot_ecdf`](#fastmob.core.base.BaseDataFrame.plot_ecdf) | Build an ECDF chart for an existing numeric or list-valued column. | [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) |
| [`plot_heatmap`](#fastmob.core.base.BaseDataFrame.plot_heatmap) | Build a long-form heatmap from existing columns. | [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) |
| [`plot_scatter`](#fastmob.core.base.BaseDataFrame.plot_scatter) | Build a scatter chart from existing columns. | [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) |
| [`to_native`](#fastmob.core.base.BaseDataFrame.to_native) | Return the stored native dataframe (a PyArrow table for tables built by the library's own readers). | [Convert and compare typed tables](../../learn/process-trajectories.md#convert-and-compare-typed-tables) |
| [`to_pandas`](#fastmob.core.base.BaseDataFrame.to_pandas) | Return the wrapped data as a native pandas DataFrame. | [Load the data](../../learn/clean-and-detect-stops.md#load-the-data) |
| [`to_polars`](#fastmob.core.base.BaseDataFrame.to_polars) | Return the wrapped data as a native Polars DataFrame. | [Convert and compare typed tables](../../learn/process-trajectories.md#convert-and-compare-typed-tables) |
<!-- api-table:end -->

::: fastmob.core.base.BaseDataFrame
    options:
      show_source: false

<!-- example-link -->**Worked example:** [Convert and compare typed tables](../../learn/process-trajectories.md#convert-and-compare-typed-tables) · [Chart the results](../../learn/analyze-a-simple-trajectory.md#chart-the-results) · [Load the data](../../learn/clean-and-detect-stops.md#load-the-data)

