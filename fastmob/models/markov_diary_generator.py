from __future__ import annotations

import datetime
import secrets

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob import _core

DATETIME = "datetime"
UID = "uid"

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
    granularity_minutes : int, optional
        Temporal resolution of a diary slot, in minutes. Must be a positive
        divisor of 1440. The default is ``60`` (hourly, 24 slots/day);
        ``15`` gives 96 slots/day and ``5`` gives 288 slots/day.

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
    >>> from fastmob.models import MarkovDiaryGenerator
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

    def __init__(self, name="Markov diary", granularity_minutes=60):
        if granularity_minutes <= 0 or 1440 % granularity_minutes != 0:
            raise ValueError("granularity_minutes must be a positive divisor of 1440")
        self._cdf_matrix_flat: pa.Array | None = None
        self._name = name
        self._granularity_minutes = int(granularity_minutes)
        self._slots_per_day = 1440 // int(granularity_minutes)
        self._time_slot_length = f"{int(granularity_minutes)}min"

    @property
    def name(self):
        return self._name

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
        if n_individuals < 0:
            raise ValueError("n_individuals must be greater than or equal to zero")

        frame = nw.from_native(getattr(traj, "df", traj), eager_only=True)
        missing = [column for column in (UID, DATETIME, lid) if column not in frame.columns]
        if missing:
            raise ValueError(f"trajectory is missing required columns: {missing}")

        uid_values = frame.get_column(UID).cast(nw.String).to_arrow()
        location_values = frame.get_column(lid).cast(nw.String).to_arrow()
        timestamps = frame.get_column(DATETIME).cast(nw.Datetime("ns")).to_arrow()
        if uid_values.null_count:
            raise ValueError(f"trajectory column {UID!r} must not contain null values")
        if timestamps.null_count:
            raise ValueError(f"trajectory column {DATETIME!r} must not contain null values")
        if location_values.null_count:
            raise ValueError(f"trajectory column {lid!r} must not contain null values")

        uid_codes = pc.cast(pc.dictionary_encode(uid_values).indices, pa.int64())
        location_codes = pc.cast(pc.dictionary_encode(location_values).indices, pa.int64())
        timestamps_ns = pc.cast(timestamps, pa.int64())
        self._cdf_matrix_flat = _core.markov_diary_fit_from_arrow(
            uid_codes,
            timestamps_ns,
            location_codes,
            int(n_individuals),
            self._slots_per_day,
        )

    def generate(self, diary_length, start_date, random_state=None):
        """Generate a synthetic mobility diary.

        Samples a sequence of hourly abstract locations from the fitted Markov chain
        and returns a compact representation where consecutive identical locations
        are collapsed to a single row.

        Parameters
        ----------
        diary_length : int
            Length of the diary in time slots (equal to hours when
            ``granularity_minutes=60``; e.g. 96 slots = one day at 15-min
            granularity).
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
        # When unfitted, pass cdf_matrix=None so the Rust core builds a home-only
        # CDF sized for this generator's granularity (agent stays home all day).
        seed = int(random_state) if random_state is not None else secrets.randbits(31)
        start_ts = int(start_date.timestamp())
        ts_arr, locs_arr, starts, ends = _core.markov_diary_batch_generate_arrow(
            self._cdf_matrix_flat, diary_length, start_ts, 1, seed, self._slots_per_day
        )
        start, end = starts[0], ends[0]
        timestamps = pa.array(
            [
                datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).replace(tzinfo=None)
                for t in ts_arr.slice(start, end - start).to_pylist()
            ],
            type=pa.timestamp("s"),
        )
        return pa.table({DATETIME: timestamps, "abstract_location": locs_arr.slice(start, end - start)})

    def _generate_list(self, diary_length, start_date, random_state=None):
        """Compatibility shim used by STS_epr — returns list of [datetime, abstract_location]."""
        return [
            [row[DATETIME], row["abstract_location"]]
            for row in self.generate(diary_length, start_date, random_state=random_state).to_pylist()
        ]
