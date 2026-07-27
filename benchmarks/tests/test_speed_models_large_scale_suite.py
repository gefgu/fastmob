from __future__ import annotations

from benchmarks import speed_models_large_scale as suite


def test_large_scale_registry_contains_fixed_sts_epr_cases():
    assert {"sts_epr_20a", "sts_epr_100a"}.issubset({spec.name for spec in suite.LARGE_SCALE_BENCHMARKS})


def test_parse_args_accepts_metrics_filter():
    args = suite.parse_args(
        [
            "--library",
            "fastmob",
            "--mode",
            "trajectory",
            "--metrics",
            "sts_epr_custom",
            "--sts-epr-agents",
            "750",
            "--sizes",
            "1000",
            "--iterations",
            "3",
        ]
    )

    assert args.metrics == ["sts_epr_custom"]
    assert args.sts_epr_agents == 750
    assert args.n_agents == [750]
    assert args.sizes == [1000]
    selected = suite.selected_specs(args)
    assert [spec.name for spec in selected] == ["sts_epr_750a"]
    assert selected[0].kwargs["n_agents"] == 750


def test_location_mode_defaults_to_location_metrics():
    args = suite.parse_args(["--library", "fastmob", "--mode", "location"])

    assert args.metrics == [spec.name for spec in suite.LOCATION_MODEL_BENCHMARKS]
    assert [spec.name for spec in suite.selected_specs(args)] == [spec.name for spec in suite.LOCATION_MODEL_BENCHMARKS]
