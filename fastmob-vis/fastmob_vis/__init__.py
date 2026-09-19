"""Generic, Narwhals-native visualization API for Fastmob."""

from .brand import PALETTES
from .charts import Chart, bar, boxplot, heatmap, map, scatter
from .charts import ecdf as _ecdf
from .figure import EChartsFigure, get_resource_bundle
from .plots import (
    plot_activity_transition_difference,
    plot_activity_transition_matrix,
    plot_daily_activity_difference,
    plot_daily_activity_distribution,
    plot_distance_frequency_law,
    plot_dwell_time_ecdf,
    plot_jump_lengths_ecdf,
    plot_lognormal_fits,
    plot_mobility_profiles,
    plot_motif_literature_comparison,
    plot_profile_metrics,
    plot_radius_of_gyration_ecdf,
    plot_stvd_comparison,
    plot_trip_duration_ecdf,
    plot_truncated_powerlaw_fits,
    plot_visit_purpose_comparison,
    plot_visit_purpose_distribution,
    plot_visits_frequency_ecdf,
)

# Loading ``.plots`` imports the ``fastmob_vis.ecdf`` module, which Python
# places on this package as ``ecdf``. Restore the documented chart constructor.
ecdf = _ecdf

__all__ = [
    "PALETTES",
    "Chart",
    "EChartsFigure",
    "bar",
    "boxplot",
    "ecdf",
    "get_resource_bundle",
    "heatmap",
    "map",
    "plot_activity_transition_difference",
    "plot_activity_transition_matrix",
    "plot_daily_activity_difference",
    "plot_daily_activity_distribution",
    "plot_distance_frequency_law",
    "plot_dwell_time_ecdf",
    "plot_jump_lengths_ecdf",
    "plot_lognormal_fits",
    "plot_mobility_profiles",
    "plot_motif_literature_comparison",
    "plot_profile_metrics",
    "plot_radius_of_gyration_ecdf",
    "plot_stvd_comparison",
    "plot_trip_duration_ecdf",
    "plot_truncated_powerlaw_fits",
    "plot_visit_purpose_comparison",
    "plot_visit_purpose_distribution",
    "plot_visits_frequency_ecdf",
    "scatter",
]
