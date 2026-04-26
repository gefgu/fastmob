from __future__ import annotations


def pytest_addoption(parser):
    group = parser.getgroup("skmob2 profiling")
    group.addoption(
        "--profile-workload",
        action="store",
        default=None,
        help="Brightkite workload name to run in profiling tests.",
    )
    group.addoption(
        "--profile-rows",
        action="store",
        default="4000000",
        help="Number of Brightkite rows to load for profiling workloads.",
    )
    group.addoption(
        "--profile-backend",
        action="store",
        default="pandas",
        choices=("pandas", "polars"),
        help="DataFrame backend to use for profiling workloads.",
    )
