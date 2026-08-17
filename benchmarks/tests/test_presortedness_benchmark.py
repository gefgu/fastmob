"""Tests for the presortedness cost benchmark helpers."""

import pandas as pd

from benchmarks.individual.presortedness_benchmark import (
    check_users_and_timestamps_sorted,
    check_users_and_timestamps_sorted_rust,
    check_users_contiguous,
    check_users_contiguous_rust,
)


def _frame(users, timestamps):
    return pd.DataFrame(
        {
            "user": users,
            "check-in_time": pd.to_datetime(timestamps),
            "latitude": [0.0] * len(users),
            "longitude": [0.0] * len(users),
        }
    )


def test_checks_accept_contiguous_users_and_monotonic_timestamps():
    frame = _frame(
        ["b", "b", "a", "a"],
        ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03"],
    )
    assert check_users_contiguous(frame)
    assert check_users_and_timestamps_sorted(frame)


def test_user_check_rejects_a_user_that_reappears():
    frame = _frame(
        ["a", "b", "a"],
        ["2020-01-01", "2020-01-01", "2020-01-02"],
    )
    assert not check_users_contiguous(frame)
    assert not check_users_and_timestamps_sorted(frame)


def test_timestamp_check_rejects_only_within_user_inversions():
    frame = _frame(
        ["a", "a", "b", "b"],
        ["2020-01-02", "2020-01-01", "2020-01-01", "2020-01-02"],
    )
    assert check_users_contiguous(frame)
    assert not check_users_and_timestamps_sorted(frame)


def test_checks_handle_empty_and_single_row_frames():
    empty = _frame([], [])
    single = _frame(["a"], ["2020-01-01"])
    for frame in (empty, single):
        assert check_users_contiguous(frame)
        assert check_users_and_timestamps_sorted(frame)


def test_rust_checks_match_python_check_semantics():
    cases = [
        _frame(["a", "a", "b", "b"], ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
        _frame(["a", "b", "a"], ["2020-01-01"] * 3),
        _frame(["a", "a"], ["2020-01-02", "2020-01-01"]),
    ]
    for frame in cases:
        assert check_users_contiguous_rust(frame) == check_users_contiguous(frame)
        assert check_users_and_timestamps_sorted_rust(frame) == check_users_and_timestamps_sorted(frame)
