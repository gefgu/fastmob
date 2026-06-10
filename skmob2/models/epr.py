from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from ._common import (
    LATITUDE,
    LONGITUDE,
    RELEVANCE,
    tessellation_lat_lngs,
    to_pandas_frame,
    trajectory_dataframe,
)
from .gravity import Gravity


def _unwrap_native_frame(df: Any) -> Any:
    return getattr(df, "df", df)


def _tessellation_arrays(
    spatial_tessellation: Any, relevance_column: str | None
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    native = _unwrap_native_frame(spatial_tessellation)
    nw_df = nw.from_native(native, eager_only=True)
    columns = set(nw_df.columns)
    if "geometry" in columns:
        pandas_frame = to_pandas_frame(spatial_tessellation)
        lats_lngs = tessellation_lat_lngs(pandas_frame)
        relevances = (
            np.ones(len(pandas_frame), dtype=float)
            if relevance_column is None
            else pandas_frame[relevance_column].fillna(0).to_numpy(dtype=float)
        )
        return pandas_frame, nw.from_native(pandas_frame, eager_only=True).implementation, lats_lngs, relevances

    if LATITUDE in columns and LONGITUDE in columns:
        lat_col, lng_col = LATITUDE, LONGITUDE
    elif "latitude" in columns and "longitude" in columns:
        lat_col, lng_col = "latitude", "longitude"
    elif "lat" in columns and "lon" in columns:
        lat_col, lng_col = "lat", "lon"
    else:
        raise ValueError("spatial_tessellation must include a geometry column or latitude/longitude columns.")

    lats = np.asarray(nw_df.get_column(lat_col).to_numpy(), dtype=float)
    lngs = np.asarray(nw_df.get_column(lng_col).to_numpy(), dtype=float)
    if relevance_column is None:
        relevances = np.ones(len(nw_df), dtype=float)
    else:
        relevances = np.asarray(nw_df.get_column(relevance_column).to_numpy(), dtype=float)
        relevances = np.nan_to_num(relevances, nan=0.0)
    return nw_df.to_native(), nw_df.implementation, np.column_stack((lats, lngs)), relevances


def _trajectory_native_frame(agent_ids: Any, lats: Any, lngs: Any, timestamps: Any, backend: Any) -> Any:
    datetime_values = np.asarray(timestamps, dtype="datetime64[s]").astype("datetime64[ms]")
    values = {
        "uid": agent_ids,
        "lat": lats,
        "lng": lngs,
        "datetime": datetime_values,
    }
    return nw.from_dict(values, backend=backend).to_native()


def compute_od_matrix(
    gravity_singly,
    spatial_tessellation,
    tile_id_column="tile_id",
    relevance_column=RELEVANCE,
):
    """Compute an OD probability matrix from a singly constrained gravity model.

    Returns a 2-D numpy array ``M`` where element ``M[i, j]`` is the probability
    :math:`p_{ij}` of moving from location ``i`` to location ``j``, as predicted
    by the singly constrained Gravity model applied to ``spatial_tessellation``.

    Parameters
    ----------
    gravity_singly : Gravity
        A :class:`Gravity` instance with ``gravity_type="singly constrained"``.
    spatial_tessellation : DataFrame or GeoDataFrame
        The spatial tessellation describing the division of the territory into
        locations.
    tile_id_column : str, optional
        Name of the column containing the location identifier. The default is
        ``"tile_id"``.
    relevance_column : str, optional
        Name of the column containing the location relevance. The default is
        ``"relevance"``.

    Returns
    -------
    numpy.ndarray
        A ``(n_locs, n_locs)`` array of trip probabilities. Each row sums to 1.
    """
    return gravity_singly.generate(
        spatial_tessellation,
        tile_id_column=tile_id_column,
        tot_outflows_column=None,
        relevance_column=relevance_column,
        out_format="probabilities",
    ).to_matrix()


class EPR:
    """Exploration and Preferential Return (EPR) trajectory generator.

    The EPR model generates synthetic individual mobility trajectories based on
    two competing mechanisms drawn from empirical observations of human travel
    patterns [SKWB2010]_:

    **Waiting time choice.** The time :math:`\\Delta t` between consecutive moves
    is drawn from a truncated power law:

    .. math::

        P(\\Delta t) \\sim \\Delta t^{-1-\\beta} \\exp(-\\Delta t / \\tau).

    **Action selection.** With probability :math:`P_{\\text{new}} = \\rho S^{-\\gamma}`,
    where :math:`S` is the number of distinct previously visited locations, the agent
    explores a new location; otherwise it returns to a previously visited one.

    **Exploration.** The new location :math:`j \\neq i` is chosen with probability
    proportional to the gravity-model score :math:`p_{ij}` derived from
    ``gravity_singly``.

    **Return.** A previously visited location :math:`i` is chosen with probability
    :math:`\\Pi_i = f_i`, proportional to its visitation frequency.

    This class is the base for :class:`DensityEPR`, :class:`SpatialEPR`, and
    :class:`Ditras`.

    Parameters
    ----------
    name : str, optional
        Human-readable label for this model instance. The default is
        ``"EPR model"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]` in
        :math:`P_{\\text{new}} = \\rho S^{-\\gamma}`. Controls the overall tendency
        to explore new locations. The default is ``0.6`` [SKWB2010]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0` in
        :math:`P_{\\text{new}} = \\rho S^{-\\gamma}`. Larger values cause agents to
        return more frequently as they accumulate visited locations. The default is
        ``0.21`` [SKWB2010]_.
    beta : float, optional
        Tail exponent :math:`\\beta` of the waiting-time distribution. The default
        is ``0.8`` [SKWB2010]_.
    tau : int, optional
        Characteristic time scale :math:`\\tau` (in hours) of the waiting-time
        distribution. The default is ``17`` [SKWB2010]_.
    min_wait_time_minutes : int, optional
        Minimum waiting time between two moves, in minutes. The default is ``20``.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.
    rho : float
        The exploration parameter :math:`\\rho`.
    gamma : float
        The return-penalty exponent :math:`\\gamma`.
    beta : float
        The waiting-time tail exponent :math:`\\beta`.
    tau : int
        The waiting-time scale :math:`\\tau` (in hours).
    min_wait_time : float
        Minimum waiting time in hours (``min_wait_time_minutes / 60``).
    spatial_tessellation_ : DataFrame or None
        The tessellation used in the last ``generate()`` call.
    trajectories_ : list
        Raw trajectory records produced by the last ``generate()`` call.

    References
    ----------
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabási, A.-L. (2010). Modelling
       the scaling properties of human mobility. *Nature Physics*, 6, 818–823.

    See Also
    --------
    DensityEPR, SpatialEPR, Ditras
    """

    def __init__(
        self,
        name="EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        self._name = name
        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta
        self._spatial_tessellation = None
        self.lats_lngs = None
        self.relevances = None
        self.gravity_singly = None
        self._min_wait_time = min_wait_time_minutes / 60.0
        self._trajectories_ = []

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def beta(self):
        return self._beta

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation

    @property
    def trajectories_(self):
        return self._trajectories_

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        """Simulate agents from ``start_date`` to ``end_date``.

        Parameters
        ----------
        start_date : pandas.Timestamp or datetime
            Start time of the simulation.
        end_date : pandas.Timestamp or datetime
            End time of the simulation.
        spatial_tessellation : DataFrame or GeoDataFrame
            Division of the territory into locations. Must include either a
            ``geometry`` column or explicit ``lat`` / ``lng`` columns.
        gravity_singly : Gravity or None, optional
            A singly constrained :class:`Gravity` model used to derive
            origin–destination trip probabilities. If ``None``, a default
            ``Gravity(gravity_type="singly constrained")`` is created automatically.
            The default is ``None``.
        n_agents : int, optional
            Number of agents to simulate. The default is ``1``.
        starting_locations : list of int or None, optional
            One starting tessellation index per agent. Length must equal
            ``n_agents``. If ``None``, starting locations are chosen uniformly
            at random. The default is ``None``.
        relevance_column : str, optional
            Name of the column in ``spatial_tessellation`` used as location
            relevance for the gravity model. The default is ``"relevance"``.
        random_state : int or None, optional
            Seed for the random number generator, enabling reproducible results.
            The default is ``None`` (non-deterministic).
        log_file : str or None, optional
            Unused in skmob2 (retained for API compatibility). The default is
            ``None``.
        show_progress : bool, optional
            Unused in skmob2 (retained for API compatibility). The default is
            ``False``.

        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``. One row per recorded position.

        Raises
        ------
        IndexError
            If ``starting_locations`` is provided and has fewer entries than
            ``n_agents``.
        TypeError
            If ``gravity_singly`` is not a :class:`Gravity` instance.
        AttributeError
            If ``gravity_singly`` is a :class:`Gravity` instance but its
            ``gravity_type`` is not ``"singly constrained"``.
        """
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")
        if gravity_singly is None:
            self.gravity_singly = Gravity(gravity_type="singly constrained")
        elif isinstance(gravity_singly, Gravity):
            if gravity_singly.gravity_type == "singly constrained":
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError(
                    "Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'."
                )
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

        parameters = {
            "model": {
                "class": self.__class__.__init__,
                "generate": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "gravity_singly": gravity_singly,
                    "n_agents": n_agents,
                    "relevance_column": relevance_column,
                    "random_state": random_state,
                    "show_progress": show_progress,
                },
            }
        }

        if random_state is not None and int(random_state) < 0:
            raise ValueError("random_state must be a non-negative integer.")

        self._trajectories_ = []
        self._spatial_tessellation, output_backend, self.lats_lngs, self.relevances = _tessellation_arrays(
            spatial_tessellation, relevance_column
        )

        start_values = None if starting_locations is None else np.asarray(starting_locations, dtype=np.int64)

        rows = self._epr_generate_parallel(
            start_date,
            end_date,
            n_agents,
            random_state=None if random_state is None else int(random_state),
            starting_locations=start_values,
            output_backend=output_backend,
        )
        return trajectory_dataframe(rows, parameters=parameters)

    def _epr_generate_parallel(
        self,
        start_date,
        end_date,
        n_agents,
        *,
        random_state=None,
        starting_locations=None,
        output_backend=None,
    ):
        from skmob2 import _core

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        lats = np.ascontiguousarray(self.lats_lngs[:, 0], dtype=float)
        lngs = np.ascontiguousarray(self.lats_lngs[:, 1], dtype=float)
        relevances = np.ascontiguousarray(self.relevances, dtype=float)
        starts = None if starting_locations is None else np.ascontiguousarray(starting_locations, dtype=np.int64)
        agent_ids, lats_out, lngs_out, timestamps = _core.model_epr_simulate_agents(
            lats,
            lngs,
            relevances,
            float(self._rho),
            float(self._gamma),
            float(self._beta),
            float(self._tau),
            float(self._min_wait_time),
            start_ts,
            end_ts,
            self.gravity_singly.deterrence_func_type,
            float(self.gravity_singly.deterrence_func_args[0]),
            float(self.gravity_singly.origin_exp),
            float(self.gravity_singly.destination_exp),
            int(n_agents),
            random_state,
            starts,
        )
        return _trajectory_native_frame(agent_ids, lats_out, lngs_out, timestamps, output_backend)


class DensityEPR(EPR):
    """Density-EPR (d-EPR) trajectory generator.

    The d-EPR model extends :class:`EPR` by weighting exploratory moves according
    to a gravity model that accounts for location relevance (e.g. population
    density) [PSRPGB2015]_ [PSR2016]_. The four mechanisms are:

    **Waiting time choice.** The waiting time :math:`\\Delta t` between two moves
    is drawn from :math:`P(\\Delta t) \\sim \\Delta t^{-1-\\beta} \\exp(-\\Delta t/\\tau)`.

    **Action selection.** With probability :math:`P_{\\text{new}} = \\rho S^{-\\gamma}`,
    where :math:`S` is the number of distinct previously visited locations, the agent
    explores a new location; otherwise it returns to a previously visited one.

    **Exploration.** If the agent at location :math:`i` explores, the destination
    :math:`j \\neq i` is selected with probability proportional to the gravity score:

    .. math::

        p_{ij} = \\frac{n_i n_j}{r_{ij}^2}

    where :math:`n_{i(j)}` is the relevance of location :math:`i(j)` and
    :math:`r_{ij}` is their Haversine distance. The set of visited locations
    :math:`S` grows by 1.

    **Return.** A previously visited location :math:`i` is chosen with probability
    proportional to its visitation frequency: :math:`\\Pi_i = f_i`.

    Parameters
    ----------
    name : str, optional
        Human-readable label. The default is ``"Density EPR model"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]`. The default is ``0.6``
        [SKWB2010]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0`. The default is ``0.21``
        [SKWB2010]_.
    beta : float, optional
        Waiting-time tail exponent :math:`\\beta`. The default is ``0.8``
        [SKWB2010]_.
    tau : int, optional
        Waiting-time scale :math:`\\tau` in hours. The default is ``17``
        [SKWB2010]_.
    min_wait_time_minutes : int, optional
        Minimum waiting time between moves, in minutes. The default is ``20``.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.
    rho : float
        The exploration parameter :math:`\\rho`.
    gamma : float
        The return-penalty exponent :math:`\\gamma`.
    beta : float
        The waiting-time tail exponent :math:`\\beta`.
    tau : int
        The waiting-time scale :math:`\\tau` (in hours).
    min_wait_time : float
        Minimum waiting time in hours.

    Notes
    -----
    ``spatial_tessellation`` accepts any eager dataframe (pandas, polars, …). If
    a ``relevance`` column is absent, all locations are treated as equally relevant.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.models import DensityEPR
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [0.40, 0.30, 0.18, 0.12],
    ... })
    >>> start = pd.Timestamp("2020-01-01 08:00:00")
    >>> end   = pd.Timestamp("2020-01-14 08:00:00")
    >>> tdf = DensityEPR().generate(start, end, tessellation, n_agents=2, random_state=0)
    >>> print(tdf.head())

    References
    ----------
    .. [PSRPGB2015] Pappalardo, L. et al. (2015). Returners and Explorers dichotomy
       in human mobility. *Nature Communications*, 6, 8166.
    .. [PSR2016] Pappalardo, L., Simini, F. & Rinzivillo, S. (2016). Human Mobility
       Modelling: exploration and preferential return meet the gravity model.
       *Procedia Computer Science*, 83.
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabási, A.-L. (2010). Modelling
       the scaling properties of human mobility. *Nature Physics*, 6, 818–823.

    See Also
    --------
    EPR, SpatialEPR, Ditras
    """

    def __init__(
        self,
        name="Density EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        super().__init__(
            rho=rho,
            gamma=gamma,
            beta=beta,
            tau=tau,
            min_wait_time_minutes=min_wait_time_minutes,
        )
        self._name = name

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        """Simulate agents from ``start_date`` to ``end_date``.

        Parameters
        ----------
        start_date : pandas.Timestamp or datetime
            Start time of the simulation.
        end_date : pandas.Timestamp or datetime
            End time of the simulation.
        spatial_tessellation : DataFrame or GeoDataFrame
            Division of the territory into locations. Must include either a
            ``geometry`` column or explicit ``lat`` / ``lng`` columns.
        gravity_singly : Gravity or None, optional
            A singly constrained :class:`Gravity` model for the exploration phase.
            If ``None``, a default ``Gravity(gravity_type="singly constrained")``
            is used. The default is ``None``.
        n_agents : int, optional
            Number of agents to simulate. The default is ``1``.
        starting_locations : list of int or None, optional
            One tessellation index per agent as the starting position. If ``None``,
            chosen uniformly at random. The default is ``None``.
        relevance_column : str, optional
            Column in ``spatial_tessellation`` used as location relevance for the
            gravity model. The default is ``"relevance"``.
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.
        log_file : str or None, optional
            Unused (retained for API compatibility). The default is ``None``.
        show_progress : bool, optional
            Unused (retained for API compatibility). The default is ``False``.

        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``.
        """
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            relevance_column=relevance_column,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class SpatialEPR(EPR):
    """Spatial-EPR (s-EPR) trajectory generator.

    The s-EPR model is a distance-only variant of :class:`DensityEPR` that ignores
    location relevance during exploration [PSRPGB2015]_ [PSR2016]_ [SKWB2010]_. The
    four mechanisms are:

    **Waiting time choice.** The waiting time :math:`\\Delta t` between two moves is
    drawn from :math:`P(\\Delta t) \\sim \\Delta t^{-1-\\beta} \\exp(-\\Delta t/\\tau)`.

    **Action selection.** With probability :math:`P_{\\text{new}} = \\rho S^{-\\gamma}`,
    where :math:`S` is the number of distinct previously visited locations, the agent
    explores a new location; otherwise it returns to a previously visited one.

    **Exploration.** If the agent at location :math:`i` explores, the unvisited
    destination :math:`j \\neq i` is selected inversely proportional to squared
    distance only:

    .. math::

        p_{ij} = \\frac{1}{r_{ij}^2}.

    **Return.** A previously visited location :math:`i` is chosen with probability
    proportional to its visitation frequency: :math:`\\Pi_i = f_i`.

    Parameters
    ----------
    name : str, optional
        Human-readable label. The default is ``"Spatial EPR model"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]`. The default is ``0.6``
        [SKWB2010]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0`. The default is ``0.21``
        [SKWB2010]_.
    beta : float, optional
        Waiting-time tail exponent :math:`\\beta`. The default is ``0.8``
        [SKWB2010]_.
    tau : int, optional
        Waiting-time scale :math:`\\tau` in hours. The default is ``17``
        [SKWB2010]_.
    min_wait_time_minutes : int, optional
        Minimum waiting time between moves, in minutes. The default is ``20``.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.
    rho : float
        The exploration parameter :math:`\\rho`.
    gamma : float
        The return-penalty exponent :math:`\\gamma`.
    beta : float
        The waiting-time tail exponent :math:`\\beta`.
    tau : int
        The waiting-time scale :math:`\\tau` (in hours).
    min_wait_time : float
        Minimum waiting time in hours.

    Notes
    -----
    ``spatial_tessellation`` accepts any eager dataframe (pandas, polars, …). The
    ``relevance_column`` parameter is accepted for API compatibility but is **not
    used** — all location relevances are treated as equal in the spatial phase.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.models import SpatialEPR
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ... })
    >>> start = pd.Timestamp("2020-01-01 08:00:00")
    >>> end   = pd.Timestamp("2020-01-14 08:00:00")
    >>> tdf = SpatialEPR().generate(start, end, tessellation, n_agents=2, random_state=0)
    >>> print(tdf.head())

    References
    ----------
    .. [PSRPGB2015] Pappalardo, L. et al. (2015). Returners and Explorers dichotomy
       in human mobility. *Nature Communications*, 6, 8166.
    .. [PSR2016] Pappalardo, L., Simini, F. & Rinzivillo, S. (2016). Human Mobility
       Modelling: exploration and preferential return meet the gravity model.
       *Procedia Computer Science*, 83.
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabási, A.-L. (2010). Modelling
       the scaling properties of human mobility. *Nature Physics*, 6, 818–823.

    See Also
    --------
    EPR, DensityEPR, Ditras
    """

    def __init__(
        self,
        name="Spatial EPR model",
        rho=0.6,
        gamma=0.21,
        beta=0.8,
        tau=17,
        min_wait_time_minutes=20,
    ):
        super().__init__(
            rho=rho,
            gamma=gamma,
            beta=beta,
            tau=tau,
            min_wait_time_minutes=min_wait_time_minutes,
        )
        self._name = name

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        """Simulate agents from ``start_date`` to ``end_date``.

        Parameters
        ----------
        start_date : pandas.Timestamp or datetime
            Start time of the simulation.
        end_date : pandas.Timestamp or datetime
            End time of the simulation.
        spatial_tessellation : DataFrame or GeoDataFrame
            Division of the territory into locations.
        gravity_singly : Gravity or None, optional
            A singly constrained :class:`Gravity` model. The relevance column is
            ignored; only distances affect exploration probabilities. If ``None``,
            a default ``Gravity(gravity_type="singly constrained")`` is created.
            The default is ``None``.
        n_agents : int, optional
            Number of agents to simulate. The default is ``1``.
        starting_locations : list of int or None, optional
            One tessellation index per agent as the starting position. The default
            is ``None`` (chosen uniformly at random).
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.
        log_file : str or None, optional
            Unused (retained for API compatibility). The default is ``None``.
        show_progress : bool, optional
            Unused (retained for API compatibility). The default is ``False``.

        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``.
        """
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            relevance_column=None,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class Ditras(EPR):
    """DITRAS (DIary-based TRAjectory Simulator) modelling framework.

    DITRAS simulates spatio-temporal mobility patterns by combining two probabilistic
    models [PS2018]_:

    **Phase 1 — Mobility Diary Generation.** A :class:`MarkovDiaryGenerator` produces
    a *mobility diary* :math:`D(t)` that encodes the temporal pattern of an agent's
    routine (home vs. away at each hour).

    **Phase 2 — Trajectory Generation.** The mobility diary drives a
    :class:`DensityEPR`-like spatial mechanism: diary slots labelled ``0`` map the
    agent back to its home location; other slots trigger an EPR exploration/return
    step to assign a physical location.

    Parameters
    ----------
    diary_generator : MarkovDiaryGenerator
        A fitted :class:`MarkovDiaryGenerator` used to produce mobility diaries.
    name : str, optional
        Human-readable label. The default is ``"Ditras model"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]` for the spatial phase.
        The default is ``0.3`` [PS2018]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0` for the spatial phase.
        The default is ``0.21`` [PS2018]_.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.
    rho : float
        The exploration parameter :math:`\\rho`.
    gamma : float
        The return-penalty exponent :math:`\\gamma`.

    Notes
    -----
    ``spatial_tessellation`` accepts any eager dataframe (pandas, polars, …). The
    diary generator must be fitted via :meth:`MarkovDiaryGenerator.fit` before
    passing it to this class.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.models import Ditras
    >>> from skmob2.models import MarkovDiaryGenerator
    >>> # Build a minimal diary generator (normally trained on real trajectories)
    >>> mdg = MarkovDiaryGenerator()
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [0.40, 0.30, 0.18, 0.12],
    ... })
    >>> start = pd.Timestamp("2020-01-01 08:00:00")
    >>> end   = pd.Timestamp("2020-01-14 08:00:00")
    >>> ditras = Ditras(mdg)
    >>> tdf = ditras.generate(start, end, tessellation,
    ...                       relevance_column="relevance", n_agents=2,
    ...                       random_state=0)
    >>> print(tdf.head())

    References
    ----------
    .. [PS2018] Pappalardo, L. & Simini, F. (2018). Data-driven generation of
       spatio-temporal routines in human mobility. *Data Mining and Knowledge
       Discovery*, 32, 787–829.

    See Also
    --------
    DensityEPR, MarkovDiaryGenerator
    """

    def __init__(self, diary_generator, name="Ditras model", rho=0.3, gamma=0.21):
        super().__init__(rho=rho, gamma=gamma)
        self._diary_generator = diary_generator
        self._name = name
        self._rho = rho
        self._gamma = gamma

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        gravity_singly=None,
        n_agents=1,
        starting_locations=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        """Simulate agents from ``start_date`` to ``end_date``.

        Parameters
        ----------
        start_date : pandas.Timestamp or datetime
            Start time of the simulation.
        end_date : pandas.Timestamp or datetime
            End time of the simulation.
        spatial_tessellation : DataFrame or GeoDataFrame
            Division of the territory into locations. Must include either a
            ``geometry`` column or explicit ``lat`` / ``lng`` columns.
        gravity_singly : Gravity or None, optional
            A singly constrained :class:`Gravity` model for the exploration phase.
            If ``None``, a default ``Gravity(gravity_type="singly constrained")``
            is used. The default is ``None``.
        n_agents : int, optional
            Number of agents to simulate. The default is ``1``.
        starting_locations : list of int or None, optional
            One tessellation index per agent as the starting (home) location. If
            ``None``, chosen uniformly at random. The default is ``None``.
        relevance_column : str, optional
            Column in ``spatial_tessellation`` used as location relevance for the
            gravity exploration phase. The default is ``"relevance"``.
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.
        log_file : str or None, optional
            Unused (retained for API compatibility). The default is ``None``.
        show_progress : bool, optional
            Unused (retained for API compatibility). The default is ``False``.

        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``.
        """
        from .markov_diary_generator import MarkovDiaryGenerator

        if not isinstance(self._diary_generator, MarkovDiaryGenerator):
            raise TypeError("diary_generator must be a MarkovDiaryGenerator.")
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")
        if gravity_singly is None:
            self.gravity_singly = Gravity(gravity_type="singly constrained")
        elif isinstance(gravity_singly, Gravity):
            if gravity_singly.gravity_type == "singly constrained":
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError(
                    "Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'."
                )
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")
        if random_state is not None and int(random_state) < 0:
            raise ValueError("random_state must be a non-negative integer.")

        parameters = {
            "model": {
                "class": self.__class__.__init__,
                "generate": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "gravity_singly": gravity_singly,
                    "n_agents": n_agents,
                    "relevance_column": relevance_column,
                    "random_state": random_state,
                    "show_progress": show_progress,
                },
            }
        }

        self._trajectories_ = []
        self._spatial_tessellation, output_backend, self.lats_lngs, self.relevances = _tessellation_arrays(
            spatial_tessellation, relevance_column
        )

        rng = np.random.default_rng(random_state)
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        diary_length = max(1, int(np.ceil((end_ts - start_ts) / 3600)))
        n_locations = len(self.lats_lngs)
        if n_locations == 0:
            raise ValueError("spatial_tessellation must contain at least one location.")
        global_relevances = np.nan_to_num(np.asarray(self.relevances, dtype=float), nan=0.0)
        global_relevances = np.where(global_relevances > 0, global_relevances, 0.0)
        global_probs = None
        if global_relevances.sum() > 0:
            global_probs = global_relevances / global_relevances.sum()
        candidate_draw_size = min(256, n_locations)

        start_values = None if starting_locations is None else np.asarray(starting_locations, dtype=np.int64)
        agent_ids: list[int] = []
        lats_out: list[float] = []
        lngs_out: list[float] = []
        timestamps: list[int] = []

        def record(agent_id: int, loc: int, ts: int) -> None:
            agent_ids.append(agent_id)
            lats_out.append(float(self.lats_lngs[loc, 0]))
            lngs_out.append(float(self.lats_lngs[loc, 1]))
            timestamps.append(ts)

        def weighted_choice(candidates: list[int], weights: np.ndarray) -> int:
            if len(candidates) == 1:
                return candidates[0]
            clean = np.nan_to_num(weights.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            total = clean.sum()
            if total <= 0:
                return int(rng.choice(candidates))
            probs = clean / total
            return int(rng.choice(candidates, p=probs))

        for agent_index in range(n_agents):
            agent_id = agent_index + 1
            if start_values is not None and agent_index < len(start_values):
                home = int(min(max(int(start_values[agent_index]), 0), n_locations - 1))
            else:
                home = int(rng.integers(0, n_locations))

            current = home
            visits = {home: 1}
            record(agent_id, current, start_ts)

            diary_seed = None if random_state is None else int(random_state) + agent_index
            diary = self._diary_generator.generate(diary_length, start_date, random_state=diary_seed)
            for _, diary_row in diary.iloc[1:].iterrows():
                move_ts = int(diary_row["datetime"].timestamp())
                if move_ts <= start_ts or move_ts >= end_ts:
                    continue
                abstract_location = int(diary_row["abstract_location"])
                if abstract_location == 0:
                    next_loc = home
                else:
                    visited_away = [loc for loc in visits if loc != home]
                    s = max(1.0, float(len(visits)))
                    explore = rng.random() < float(self._rho) * s ** (-float(self._gamma))
                    sampled = rng.choice(
                        n_locations,
                        size=candidate_draw_size,
                        replace=False,
                        p=global_probs,
                    )
                    unvisited = [
                        int(loc)
                        for loc in sampled
                        if int(loc) not in visits and int(loc) != current and int(loc) != home
                    ]
                    if explore and unvisited:
                        weights = np.asarray(self.relevances[unvisited], dtype=float)
                        next_loc = weighted_choice(unvisited, weights)
                    elif visited_away:
                        weights = np.array([visits[loc] for loc in visited_away], dtype=float)
                        next_loc = weighted_choice(visited_away, weights)
                    elif unvisited:
                        weights = np.asarray(self.relevances[unvisited], dtype=float)
                        next_loc = weighted_choice(unvisited, weights)
                    else:
                        next_loc = current

                current = int(next_loc)
                visits[current] = visits.get(current, 0) + 1
                record(agent_id, current, move_ts)

        order = np.lexsort((np.asarray(timestamps), np.asarray(agent_ids)))
        rows = _trajectory_native_frame(
            np.asarray(agent_ids, dtype=np.int64)[order],
            np.asarray(lats_out, dtype=float)[order],
            np.asarray(lngs_out, dtype=float)[order],
            np.asarray(timestamps, dtype=np.int64)[order],
            output_backend,
        )
        return trajectory_dataframe(rows, parameters=parameters)
