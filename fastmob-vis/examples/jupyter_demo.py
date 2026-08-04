from fastmob_vis import (
    plot_distance_frequency_law,
    plot_jump_lengths_ecdf,
    plot_lognormal_fits,
    plot_stvd_comparison,
    plot_truncated_powerlaw_fits,
)

observed = [0.4, 1.2, 1.2, 3.8, 10.0]
synthetic = [0.6, 1.0, 2.0, 2.0, 7.5]

figure = plot_jump_lengths_ecdf(observed, synthetic, labels=("observed", "synthetic"))

powerlaw = plot_truncated_powerlaw_fits(
    ((1.2, 1.5, 1.75, 400.0), [1, 3, 10, 30], [0.18, 0.05, 0.009, 0.001], "observed"),
    ((1.0, 1.2, 1.65, 350.0), [1, 3, 10, 30], [0.16, 0.06, 0.012, 0.002], "synthetic"),
)

lognormal = plot_lognormal_fits(
    ([1, 2, 3, 5, 8], [0.24, 0.31, 0.20, 0.08, 0.02], 1.0, 0.6, "observed"),
    ([1, 2, 3, 5, 8], [0.20, 0.29, 0.22, 0.10, 0.03], 1.1, 0.65, "synthetic"),
)

distance_frequency = plot_distance_frequency_law(
    ([1, 10, 100], [10, 0.1, 0.001], 2.0, 10.0, "observed"),
    ([1, 10, 100], [8, 0.12, 0.002], 1.8, 8.0, "synthetic"),
)


def stvd_feature(area, west, south, volume_diff, peak_shift):
    return {
        "type": "Feature",
        "properties": {
            "area": area,
            "volume_diff_pct": volume_diff,
            "peak_shift_hours": peak_shift,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [west, south],
                    [west + 0.02, south],
                    [west + 0.02, south + 0.02],
                    [west, south + 0.02],
                    [west, south],
                ]
            ],
        },
    }


stvd_layers = {
    resolution: {
        "type": "FeatureCollection",
        "features": [
            stvd_feature(f"cell-{resolution}-a", 2.32, 48.84, -4.5, 1.0),
            stvd_feature(f"cell-{resolution}-b", 2.35, 48.86, 0.5, 4.0),
            stvd_feature(f"cell-{resolution}-c", 2.38, 48.88, 5.0, 8.0),
        ],
    }
    for resolution in (5, 7, 9)
}

stvd = plot_stvd_comparison(stvd_layers, center=(2.36, 48.87), zoom=10)
