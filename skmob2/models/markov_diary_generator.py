from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from skmob2 import _core

from ._common import DATETIME, UID, to_pandas_frame

_N_STATES = 48  # 24 hours × 2 typicality values


class MarkovDiaryGenerator:
    """Markov Diary Learner and Generator.

    A *Mobility Diary Learner* (MDL) learns hourly temporal mobility patterns from
    real individual trajectories and then synthesises new *mobility diaries* [PS2018]_.
    A diary :math:`D(t)` is a sequence of *abstract locations* (``0`` = home,
    ``1, 2, …`` = non-home destinations in decreasing visitation-frequency order)
    at hourly resolution.

    The learner translates each individual's trajectory into an abstract time series
    and builds a Markov chain :math:`MD(t)` over 48 states: 24 hours of the day
    :math:`\\times` 2 typicality values (1 = at the typical/home location, 0 = away).
    Transition probabilities are estimated by counting state-to-state transitions
    across all individuals.

    A fitted ``MarkovDiaryGenerator`` can be passed to :class:`Ditras` or
    :class:`STS_epr` to drive their temporal pattern.

    Parameters
    ----------
    name : str, optional
        Human-readable label for this instance. The default is ``"Markov diary"``.

    Attributes
    ----------
    name : str
        Human-readable label of this instance.
    markov_chain_ : numpy.ndarray or None
        Flattened CDF matrix built after :meth:`fit`. Shape ``(48 * 48,)``.
        ``None`` before fitting.
    time_slot_length : str
        Length of each time slot (fixed at ``"1h"``).

    Notes
    -----
    The ``fit`` method accepts any pandas-compatible dataframe. The trajectory
    must contain a datetime column (auto-detected) and a user-ID column (``uid``
    by default).

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.models import MarkovDiaryGenerator
    >>> # Normally you would pass a real trajectory TrajDataFrame here.
    >>> # This example uses an unfitted generator (home-only diary).
    >>> mdg = MarkovDiaryGenerator()
    >>> diary = mdg.generate(72, pd.Timestamp("2020-01-01 00:00:00"), random_state=0)
    >>> print(diary.head())

    References
    ----------
    .. [PS2018] Pappalardo, L. & Simini, F. (2018). Data-driven generation of
       spatio-temporal routines in human mobility. *Data Mining and Knowledge
       Discovery*, 32, 787–829.

    See Also
    --------
    Ditras, STS_epr
    """

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
        loc2rank = {loc: i + 1 for i, (loc, _) in enumerate(sorted(loc2freq.items(), key=lambda x: -x[1]))}

        bins = dt_series.dt.floor(time_slot_length)
        counts = pd.DataFrame({"bin": bins, "loc": loc_series}).groupby(["bin", "loc"]).size().reset_index(name="n")
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
        """Learn the Markov mobility diary from real trajectories.

        Builds a 48-state Markov chain from the first ``n_individuals`` users in
        ``traj``. Each state encodes an hour of the day (0–23) and a typicality
        value (1 = home, 0 = away). The resulting transition probabilities are
        stored in ``markov_chain_``.

        Parameters
        ----------
        traj : DataFrame
            Mobility trajectories. Must contain a datetime column and a ``uid``
            column (auto-detected), plus the location column specified by ``lid``.
        n_individuals : int
            Number of individuals from ``traj`` to use for training.
        lid : str, optional
            Name of the column containing the location cluster identifier. The
            default is ``"location"``.

        Notes
        -----
        Modifies the internal Markov chain in-place. Call before passing this
        instance to :class:`Ditras` or :class:`STS_epr`.
        """
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
        """Generate a synthetic mobility diary.

        Samples a sequence of hourly abstract locations from the fitted Markov chain
        and returns a compact representation where consecutive identical locations
        are collapsed to a single row.

        Parameters
        ----------
        diary_length : int
            Length of the diary in hours.
        start_date : pandas.Timestamp or datetime
            Starting timestamp for the diary.
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.

        Returns
        -------
        pandas.DataFrame
            A dataframe with columns:

            - ``datetime`` — timestamp of each location change.
            - ``abstract_location`` — abstract location identifier (``0`` = home,
              ``1, 2, …`` = non-home locations in order of decreasing frequency as
              learned from real data).

        Notes
        -----
        If :meth:`fit` has not been called, a trivial diary is generated where the
        agent stays at home (``abstract_location == 0``) for the entire period.
        """
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
