"""Generic, dataframe-shaped visualizations for a Jupyter notebook."""

import pyarrow as pa
from fastmob_vis import bar, boxplot, ecdf, heatmap, scatter

first = pa.table({"jump_lengths": [[0.4, 1.2, 1.2, 3.8, 10.0]]})
second = pa.table({"jump_lengths": [[0.6, 1.0, 2.0, 2.0, 7.5]]})
ecdf_chart = ecdf(first, value_col="jump_lengths", second=second, labels=("observed", "synthetic"))

activity = pa.table({"purpose": ["HOME", "WORK", "OTHER"], "percentage": [55.0, 35.0, 10.0]})
activity_chart = bar(activity, category_col="purpose", value_col="percentage", title="Visit purpose")

profile_points = pa.table({
    "degree_of_return": [0.2, 0.5, 0.8], "intermittency": [0.7, 0.4, 0.1],
    "profile": ["Scouter", "Regular", "Routiner"],
})
profiles_chart = scatter(profile_points, x_col="degree_of_return", y_col="intermittency", series_col="profile")

matrix = pa.table({"from": ["HOME", "HOME", "WORK", "WORK"], "to": ["HOME", "WORK", "HOME", "WORK"], "percentage": [50.0, 50.0, 30.0, 70.0]})
heatmap_chart = heatmap(matrix, x_col="to", y_col="from", value_col="percentage")

metric_values = pa.table({"profile": ["Scouter", "Scouter", "Regular", "Regular"], "regularity": [0.2, 0.3, 0.7, 0.8]})
metrics_chart = boxplot(metric_values, category_col="profile", value_col="regularity")
