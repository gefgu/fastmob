from __future__ import annotations

import datetime
import operator
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ._common import DATETIME, UID, to_pandas_frame


class MarkovDiaryGenerator:
    """Markov Diary Learner and Generator compatible with scikit-mobility."""

    def __init__(self, name="Markov diary"):
        self._markov_chain_ = None
        self._time_slot_length = "1h"
        self._name = name

    @property
    def markov_chain_(self):
        return self._markov_chain_

    @property
    def time_slot_length(self):
        return self._time_slot_length

    @property
    def name(self):
        return self._name

    def _create_empty_markov_chain(self):
        self._markov_chain_ = defaultdict(lambda: defaultdict(float))
        for h1 in range(24):
            for r1 in [0, 1]:
                for h2 in range(24):
                    for r2 in [0, 1]:
                        self._markov_chain_[(h1, r1)][(h2, r2)] = 0.0

    @staticmethod
    def _select_loc(individual_df, location2frequency, location_column="location"):
        value = individual_df[location_column]
        if isinstance(value, str):
            locations = value.split(",")
            if len(locations) == 1:
                return locations[0]
            if len(set(Counter(locations).values())) == 1:
                return sorted(
                    {k: location2frequency[k] for k in locations}.items(), key=operator.itemgetter(1), reverse=True
                )[0][0]
            return sorted(Counter(locations).items(), key=operator.itemgetter(1), reverse=True)[0][0]
        return np.nan

    @staticmethod
    def _get_location2frequency(traj, location_column="location"):
        location2frequency, location2rank = defaultdict(int), defaultdict(int)
        for _, row in traj.iterrows():
            if isinstance(row[location_column], str):
                for location in row[location_column].split(","):
                    location2frequency[location] += 1
        rank = 1
        for location, _freq in sorted(location2frequency.items(), key=operator.itemgetter(1), reverse=True):
            location2rank[location] = rank
            rank += 1
        return location2frequency, location2rank

    def _create_time_series(self, traj, lid="location"):
        traj = to_pandas_frame(traj)
        shift = traj[DATETIME].min().hour
        dt_series = traj[DATETIME]
        loc_series = traj[lid].astype("str")

        # Global location frequency — equivalent to original comma-string counting
        loc2freq = loc_series.value_counts().to_dict()
        loc2rank = {loc: i + 1 for i, (loc, _) in enumerate(sorted(loc2freq.items(), key=lambda x: -x[1]))}

        # Count per (hourly-bin, location) in one vectorized groupby
        bins = dt_series.dt.floor(self._time_slot_length)
        counts = (
            pd.DataFrame({"bin": bins, "loc": loc_series})
            .groupby(["bin", "loc"])
            .size()
            .reset_index(name="n")
        )
        # Pick best location per bin: most occurrences, tie-broken by global frequency
        counts["tiebreak"] = counts["loc"].map(loc2freq).fillna(0)
        counts.sort_values(["bin", "n", "tiebreak"], ascending=[True, False, False], inplace=True)
        best = counts.groupby("bin")["loc"].first()

        # Full hourly grid with gap-filling
        full_idx = pd.date_range(
            dt_series.min().floor(self._time_slot_length),
            dt_series.max().floor(self._time_slot_length),
            freq=self._time_slot_length,
        )
        time_series = best.reindex(full_idx).ffill().bfill().map(loc2rank)
        return time_series, shift

    def _update_markov_chain(self, time_series, shift=0):
        home = 1
        typical, non_typical = 1, 0
        values = list(time_series)
        n = len(values)
        slot = 0
        while slot < n - 1:
            h = (slot + shift) % 24
            next_h = (h + 1) % 24
            loc_h = values[slot]
            next_loc_h = values[slot + 1]
            if loc_h == home:
                if next_loc_h == home:
                    self._markov_chain_[(h, typical)][(next_h, typical)] += 1
                else:
                    tau = 1
                    if slot + 2 < n:
                        for j in range(slot + 2, n):
                            if values[j] == next_loc_h:
                                tau += 1
                            else:
                                break
                        self._markov_chain_[(h, typical)][((h + tau) % 24, non_typical)] += 1
                        slot = j - 2
                    else:
                        slot = n
            else:
                if next_loc_h == home:
                    self._markov_chain_[(h, non_typical)][(next_h, typical)] += 1
                else:
                    tau = 1
                    if slot + 2 < n:
                        for j in range(slot + 2, n):
                            if values[j] == next_loc_h:
                                tau += 1
                            else:
                                break
                        self._markov_chain_[(h, non_typical)][((h + tau) % 24, non_typical)] += 1
                        slot = j - 2
                    else:
                        slot = n
            slot += 1

    def _normalize_markov_chain(self):
        for state1 in self._markov_chain_:
            total = sum(self._markov_chain_[state1].values())
            if total != 0.0:
                for state2 in self._markov_chain_[state1]:
                    self._markov_chain_[state1][state2] /= total

    def fit(self, traj, n_individuals, lid="location"):
        traj = to_pandas_frame(traj)
        self._create_empty_markov_chain()
        individuals = traj[UID].unique()
        for individual in individuals[:n_individuals]:
            time_series, shift = self._create_time_series(traj[traj[UID] == individual], lid=lid)
            self._update_markov_chain(time_series, shift)
        self._normalize_markov_chain()

    @staticmethod
    def _weighted_random_selection(weights):
        return np.searchsorted(np.cumsum(weights), random.random())

    def _generate_list(self, diary_length, start_date, random_state=None):
        """Core generation logic; returns a plain list of [datetime, abstract_location] pairs."""
        if self._markov_chain_ is None:
            self._create_empty_markov_chain()
            for state in list(self._markov_chain_):
                h, r = state
                self._markov_chain_[state][((h + 1) % 24, r)] = 1.0
        current_date = start_date
        V, i = [], 0
        prev_state = (i, 1)
        V.append(prev_state)
        if random_state is not None:
            random.seed(random_state)
        while i < diary_length:
            h = i % 24
            p = list(self._markov_chain_[prev_state].values())
            if sum(p) == 0.0:
                hh, rr = prev_state
                next_state = ((hh + 1) % 24, rr)
            else:
                index = self._weighted_random_selection(p)
                next_state = list(self._markov_chain_[prev_state].keys())[index]
            V.append(next_state)
            j = next_state[0]
            i += j - h if j > h else 24 - h + j
            prev_state = next_state

        prev, diary, other_count = V[0], [], 1
        diary.append([current_date, 0])
        for v in V[1:]:
            h, s = v
            h_prev, _s_prev = prev
            if s == 1:
                current_date += datetime.timedelta(hours=1)
                diary.append([current_date, 0])
                other_count = 1
            else:
                j = h - h_prev if h > h_prev else 24 - h_prev + h
                for _ in range(j):
                    current_date += datetime.timedelta(hours=1)
                    diary.append([current_date, other_count])
                other_count += 1
            prev = v

        short_diary = []
        prev_location = -1
        for visit_date, abstract_location in diary[:diary_length]:
            if abstract_location != prev_location:
                short_diary.append([visit_date, abstract_location])
            prev_location = abstract_location
        return short_diary

    def generate(self, diary_length, start_date, random_state=None):
        short_diary = self._generate_list(diary_length, start_date, random_state)
        return pd.DataFrame(short_diary, columns=[DATETIME, "abstract_location"])
