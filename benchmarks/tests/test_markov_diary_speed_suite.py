from __future__ import annotations

import pytest

from benchmarks import markov_diary_speed_suite as suite


def test_parse_args_defaults_to_500_agents():
    args = suite.parse_args(["--output-dir", "/tmp/skmob2-markov-diary-test"])

    assert args.n_agents == [500]
    assert args.diary_length == 24
    assert args.fit_individuals == 3
    assert args.metrics == ["generate_public"]


def test_parse_args_accepts_multiple_agent_counts_and_fit_metric():
    args = suite.parse_args(
        [
            "--n-agents",
            "100",
            "500",
            "--diary-length",
            "168",
            "--fit-individuals",
            "5",
            "--metrics",
            "fit",
            "generate_public",
            "--output-dir",
            "/tmp/skmob2-markov-diary-test",
        ]
    )

    assert args.n_agents == [100, 500]
    assert args.diary_length == 168
    assert args.fit_individuals == 5
    assert args.metrics == ["fit", "generate_public"]


def test_summarize_times_handles_values_and_empty_list():
    assert suite.summarize_times([0.3, 0.1, 0.2]) == {
        "average_seconds": pytest.approx(0.2),
        "minimum_seconds": 0.1,
    }
    assert suite.summarize_times([]) == {"average_seconds": None, "minimum_seconds": None}
