"""Legacy convenience wrappers with generalized column auto-detection.

These wrappers are not re-exported from the top-level fastmob package.
They delegate to the canonical comparison API with None defaults so that
auto-detection from the standard column candidates applies.
"""

from __future__ import annotations

from typing import Any, Literal

from .activity import (
    activity_distribution_jensen_shannon_divergence,
    activity_transition_matrix_jensen_shannon_divergence,
)
from .distribution import (
    column_distribution_jensen_shannon_divergence,
    column_distribution_wasserstein_distance,
    visits_per_user_jensen_shannon_divergence,
    visits_per_user_wasserstein_distance,
)
from .metrics import time_bin_matrix_jensen_shannon_divergence
from .spatial import (
    od_matrix_common_part_of_commuters,
    profile_metric_wasserstein_distance,
    radius_of_gyration_wasserstein_distance,
)


def _wasserstein_column_wrapper(metric_col: str):
    def wrapper(df1: Any, df2: Any, metric_column: str = metric_col) -> float:
        return profile_metric_wasserstein_distance(df1, df2, metric_column)

    return wrapper


compare_activity_distributions = activity_distribution_jensen_shannon_divergence

compare_activity_10min_matrix = time_bin_matrix_jensen_shannon_divergence
compare_activity_transition_matrix = activity_transition_matrix_jensen_shannon_divergence


def compare_motif_distributions_with_jsd(
    agent_visitation_df: Any,
    sample_visitation_df: Any,
    user_id_col_agent: str | None = "agent_id",
    user_id_col_sample: str | None = "user_id",
    location_id_col: str | None = "area",
) -> float:
    from fastmob.measures.individual.motifs import daily_motifs

    from .activity import motif_distribution_jensen_shannon_divergence

    daily_agent = daily_motifs(agent_visitation_df, uid_col=user_id_col_agent, location_col=location_id_col)
    daily_sample = daily_motifs(sample_visitation_df, uid_col=user_id_col_sample, location_col=location_id_col)
    return motif_distribution_jensen_shannon_divergence(daily_agent, daily_sample)


compare_regularity_with_wasserstein = _wasserstein_column_wrapper("regularity")
compare_stationarity_with_wasserstein = _wasserstein_column_wrapper("stationarity")
compare_diversity_with_wasserstein = _wasserstein_column_wrapper("diversity")
compare_entropy_with_wasserstein = _wasserstein_column_wrapper("entropy")
compare_predictability_with_wasserstein = _wasserstein_column_wrapper("predictability")
compute_cpc_of_od_matrix = od_matrix_common_part_of_commuters


def compare_distributions_with_js_divergence(
    df1: Any,
    df2: Any,
    column_to_compare: str,
    bin_size: float = 1.0,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
    skip_day_period_creation: bool = False,
):
    return column_distribution_jensen_shannon_divergence(
        df1,
        df2,
        column_to_compare,
        bin_size=bin_size,
        hue=hue,
        trip_start_col1=trip_start_column,
        trip_start_col2=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_visits_per_user_with_js_divergence(
    df1: Any,
    df2: Any,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    user_id_column1: str | None = None,
    user_id_column2: str | None = None,
    bin_size: float = 1.0,
    skip_day_period_creation: bool = False,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
):
    return visits_per_user_jensen_shannon_divergence(
        df1,
        df2,
        hue=hue,
        user_id_col1=user_id_column1,
        user_id_col2=user_id_column2,
        bin_size=bin_size,
        trip_start_col=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_distributions_with_wasserstein(
    df1: Any,
    df2: Any,
    column_to_compare: str,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
    skip_day_period_creation: bool = False,
):
    return column_distribution_wasserstein_distance(
        df1,
        df2,
        column_to_compare,
        hue=hue,
        trip_start_col1=trip_start_column,
        trip_start_col2=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_visits_per_user_with_wasserstein(
    df1: Any,
    df2: Any,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    user_id_column1: str | None = None,
    user_id_column2: str | None = None,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
    skip_day_period_creation: bool = False,
):
    return visits_per_user_wasserstein_distance(
        df1,
        df2,
        hue=hue,
        user_id_col1=user_id_column1,
        user_id_col2=user_id_column2,
        trip_start_col=trip_start_column,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
        skip_day_period_creation=skip_day_period_creation,
    )


def compare_trip_length_with_wasserstein(
    trips_df1: Any,
    trips_df2: Any,
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
):
    return compare_distributions_with_wasserstein(
        trips_df1,
        trips_df2,
        "trip_length_km",
        hue=hue,
        trip_start_column=trip_start_column,
        day_column1=day_column1,
        day_column2=day_column2,
        purpose_column=purpose_column,
    )


def compare_trip_duration_with_wasserstein(
    trips_df1: Any,
    trips_df2: Any,
    duration_column: str = "Duration",
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    trip_start_column: str | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
):
    return compare_distributions_with_wasserstein(
        trips_df1,
        trips_df2,
        duration_column,
        hue=hue,
        trip_start_column=trip_start_column,
        day_column1=day_column1,
        day_column2=day_column2,
        purpose_column=purpose_column,
    )


def compare_dwell_time_with_wasserstein(
    visitation_df1: Any,
    visitation_df2: Any,
    duration_column: str = "duration_minutes",
    hue: Literal["day_of_week", "day_period", "purpose"] | None = None,
    day_column1: str | None = None,
    day_column2: str | None = None,
    purpose_column: str | None = None,
):
    from .spatial import dwell_time_wasserstein_distance

    return dwell_time_wasserstein_distance(
        visitation_df1,
        visitation_df2,
        duration_col=duration_column,
        hue=hue,
        day_col1=day_column1,
        day_col2=day_column2,
        purpose_col=purpose_column,
    )


def compare_radius_of_gyration_with_wasserstein(
    rg_df1: Any,
    rg_df2: Any,
    radius_column: str = "radius_of_gyration_km",
    grouping_column: str | None = None,
):
    return radius_of_gyration_wasserstein_distance(
        rg_df1,
        rg_df2,
        radius_col=radius_column,
        grouping_col=grouping_column,
    )
