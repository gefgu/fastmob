from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from skmob2 import _core

from ._common import DATETIME, UID, to_pandas_frame

_N_STATES = 48  # 24 hours × 2 typicality values


class MarkovDiaryGenerator:
    """Markov Diary Learner and Generator compatible with scikit-mobility."""

    def __init__(self, name="Markov diary"):
        self._cdf_matrix_flat: np.ndarray | None = None
        self._name = name

    @property
    def name(self):
        return self._name

    @staticmethod
    def _create_time_series(traj, lid="location", time_slot_length="1h"):
        """Return (values_array, shift) for one individual's trajectory.

        values_array: numpy int64 array of hourly location ranks (rank 1 = home).
        shift: int — starting hour of the first slot.
        """
        dt_series = traj[DATETIME]
        loc_series = traj[lid].astype("str")
        shift = int(dt_series.min().hour)

        loc2freq = loc_series.value_counts().to_dict()
        loc2rank = {
            loc: i + 1
            for i, (loc, _) in enumerate(sorted(loc2freq.items(), key=lambda x: -x[1]))
        }

        bins = dt_series.dt.floor(time_slot_length)
        counts = (
            pd.DataFrame({"bin": bins, "loc": loc_series})
            .groupby(["bin", "loc"])
            .size()
            .reset_index(name="n")
        )
        counts["tiebreak"] = counts["loc"].map(loc2freq).fillna(0)
        counts.sort_values(["bin", "n", "tiebreak"], ascending=[True, False, False], inplace=True)
        best = counts.groupby("bin")["loc"].first()

        full_idx = pd.date_range(
            dt_series.min().floor(time_slot_length),
            dt_series.max().floor(time_slot_length),
            freq=time_slot_length,
        )
        time_series = best.reindex(full_idx).ffill().bfill().map(loc2rank)
        values = np.array([v if not pd.isna(v) else 0 for v in time_series.values], dtype=np.uint64)
        return values, shift

    def fit(self, traj, n_individuals, lid="location"):
        traj = to_pandas_frame(traj)
        counts = np.zeros(_N_STATES * _N_STATES, dtype=np.float64)
        individuals = traj[UID].unique()
        for individual in individuals[:n_individuals]:
            ind_df = traj[traj[UID] == individual]
            values, shift = self._create_time_series(ind_df, lid=lid)
            counts = _core.markov_diary_update_chain(values, shift, counts)
        probs = _core.markov_diary_normalize(counts)
        self._cdf_matrix_flat = _core.markov_diary_build_cdf(probs)

    def generate(self, diary_length, start_date, random_state=None):
        if self._cdf_matrix_flat is None:
            # unfitted — build identity-like CDF (always advance 1 hour, stay home)
            probs = np.zeros(_N_STATES * _N_STATES, dtype=np.float64)
            for h in range(24):
                next_h = (h + 1) % 24
                probs[(h * 2 + 1) * _N_STATES + (next_h * 2 + 1)] = 1.0
            self._cdf_matrix_flat = _core.markov_diary_build_cdf(probs)

        seed = int(random_state) if random_state is not None else int(np.random.randint(0, 2**31))
        start_ts = int(start_date.timestamp())
        ts_arr, locs_arr, starts, ends = _core.markov_diary_batch_generate(
            self._cdf_matrix_flat, diary_length, start_ts, 1, seed
        )
        timestamps = [
            datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).replace(tzinfo=None)
            for t in ts_arr[starts[0] : ends[0]]
        ]
        locs = list(locs_arr[starts[0] : ends[0]])
        return pd.DataFrame({DATETIME: timestamps, "abstract_location": locs})

    def _generate_list(self, diary_length, start_date, random_state=None):
        """Compatibility shim used by STS_epr — returns list of [datetime, abstract_location]."""
        df = self.generate(diary_length, start_date, random_state=random_state)
        return [[row[DATETIME], row["abstract_location"]] for _, row in df.iterrows()]
