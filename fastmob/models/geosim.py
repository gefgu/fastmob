from __future__ import annotations

import numpy as np

from fastmob import _core

from ._common import tessellation_lat_lngs, to_pandas_frame, trajectory_dataframe


def _igraph_to_csr(
    social_graph,
    n_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a CSR adjacency list from an igraph Graph object."""
    srcs, dsts = [], []
    for edge in social_graph.es:
        srcs.append(edge.source)
        dsts.append(edge.target)
    if srcs:
        ns, nb = _core.model_social_graph_edges_to_csr(
            np.asarray(srcs, dtype=np.int64),
            np.asarray(dsts, dtype=np.int64),
            n_agents,
        )
    else:
        ns = list(range(n_agents + 1))
        nb = []
    return np.asarray(ns, dtype=np.int64), np.asarray(nb, dtype=np.int64)


class GeoSim:
    """GeoSim social trajectory generator.

    GeoSim generates synthetic individual mobility trajectories by combining the EPR
    exploration/return mechanism with a social influence layer [THSG2015]_. Each
    agent is embedded in a social network, and with probability :math:`\\alpha` it
    consults a social contact when choosing its next location.

    **Waiting time choice.** The time :math:`\\Delta t` between moves is drawn from:

    .. math::

        P(\\Delta t) \\sim \\Delta t^{-1-\\beta} \\exp(-\\Delta t/\\tau).

    **Action selection.** With probability
    :math:`P_{\\text{exp}} = \\rho S^{-\\gamma}`, where :math:`S` is the number of
    distinct previously visited locations, the agent explores a new location;
    otherwise it returns. At that point, with probability :math:`\\alpha` a social
    contact influences the location choice (Social); with probability
    :math:`1 - \\alpha` the agent acts alone (Individual).

    **Individual Exploration.** An unvisited location is selected uniformly at
    random from the set :math:`\\text{exp}(a)` of all unvisited locations for agent
    :math:`a`. The set :math:`S` grows by 1.

    **Social Exploration.** A social contact :math:`c` is selected with probability
    proportional to the cosine mobility similarity between :math:`a` and :math:`c`.
    A location is then selected from :math:`\\text{exp}(a) \\cap \\text{ret}(c)` with
    probability proportional to :math:`c`'s visitation frequency. The set :math:`S`
    grows by 1.

    **Individual Return.** A previously visited location :math:`j \\in \\text{ret}(a)`
    is selected with probability proportional to its visitation frequency
    :math:`\\Pi_j = f_j`.

    **Social Return.** A social contact :math:`c` is selected (as in Social
    Exploration). A location from :math:`\\text{ret}(a) \\cap \\text{ret}(c)` is
    selected proportionally to :math:`c`'s visitation frequency.

    Parameters
    ----------
    name : str, optional
        Human-readable label. The default is ``"GeoSim"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]`. The default is ``0.6``
        [SKWB2010]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0`. The default is ``0.21``
        [SKWB2010]_.
    alpha : float, optional
        Social influence probability :math:`\\alpha \\in [0, 1]`. The default is
        ``0.2`` [THSG2015]_.
    beta : float, optional
        Waiting-time tail exponent :math:`\\beta`. The default is ``0.8``
        [SKWB2010]_.
    tau : int, optional
        Waiting-time scale :math:`\\tau` in hours. The default is ``17``
        [SKWB2010]_.
    min_wait_time_hours : float, optional
        Minimum waiting time between moves, in hours. The default is ``1``.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.
    rho : float
        The exploration parameter :math:`\\rho`.
    gamma : float
        The return-penalty exponent :math:`\\gamma`.
    alpha : float
        The social influence probability :math:`\\alpha`.
    beta : float
        The waiting-time tail exponent :math:`\\beta`.
    tau : int
        The waiting-time scale :math:`\\tau` (in hours).
    min_wait_time_hours : float
        Minimum waiting time in hours.

    Notes
    -----
    ``spatial_tessellation`` accepts any pandas-compatible eager dataframe. It must
    contain either a ``geometry`` column or explicit ``lat`` / ``lng`` columns.
    When ``social_graph="random"`` a random geometric graph is generated
    automatically; alternatively, pass an edge list to define specific social
    connections.

    References
    ----------
    .. [PSRPGB2015] Pappalardo, L. et al. (2015). Returners and Explorers dichotomy
       in human mobility. *Nature Communications*, 6, 8166.
    .. [PSR2016] Pappalardo, L., Simini, F. & Rinzivillo, S. (2016). Human Mobility
       Modelling: exploration and preferential return meet the gravity model.
       *Procedia Computer Science*, 83.
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabási, A.-L. (2010). Modelling
       the scaling properties of human mobility. *Nature Physics*, 6, 818–823.
    .. [THSG2015] Toole, J. et al. (2015). Coupling Human Mobility and Social Ties.
       *Journal of the Royal Society Interface*, 12, 20141128.

    See Also
    --------
    EPR, SpatialEPR, Ditras, STS_epr
    """

    def __init__(
        self,
        name="GeoSim",
        rho=0.6,
        gamma=0.21,
        alpha=0.2,
        beta=0.8,
        tau=17,
        min_wait_time_hours=1,
    ):
        self.name = name
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.min_wait_time_hours = min_wait_time_hours

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        social_graph="random",
        n_agents=500,
        dt_update_mobSim=24 * 7,
        indipendency_window=0.5,
        random_state=None,
        log_file=None,
        verbose=0,
        show_progress=False,
    ):
        """Simulate a socially connected population from ``start_date`` to ``end_date``.

        Parameters
        ----------
        start_date : pandas.Timestamp or datetime
            Start time of the simulation.
        end_date : pandas.Timestamp or datetime
            End time of the simulation.
        spatial_tessellation : DataFrame or GeoDataFrame
            Division of the territory into locations. Must include either a
            ``geometry`` column or explicit ``lat`` / ``lng`` columns. Must contain
            at least 2 tiles.
        social_graph : ``"random"`` or list of (uid, uid) tuples, optional
            Social network for the simulated agents.

            - ``"random"`` — a random geometric graph on ``n_agents`` nodes is
              generated automatically.
            - A list of ``(uid_a, uid_b)`` edge tuples — the number of agents is
              inferred from the unique node set; ``n_agents`` is ignored.

            The default is ``"random"``.
        n_agents : int, optional
            Number of agents when ``social_graph="random"``. The default is ``500``.
        dt_update_mobSim : float, optional
            Interval in hours between social-graph mobility-similarity updates.
            The default is ``168`` (one week).
        indipendency_window : float, optional
            Time window in hours that must elapse before an agent's move can affect
            other agents' social choices. The default is ``0.5``.
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.
        log_file : str or None, optional
            Unused (retained for API compatibility). The default is ``None``.
        verbose : int, optional
            Unused (retained for API compatibility). The default is ``0``.
        show_progress : bool, optional
            Unused (retained for API compatibility). The default is ``False``.

        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``.

        Raises
        ------
        ValueError
            If ``n_agents <= 0``, ``start_date > end_date``,
            ``social_graph`` is an empty list, or ``social_graph`` is a string other
            than ``"random"``.
        TypeError
            If ``social_graph`` is neither a string nor a list.
        """
        if n_agents <= 0:
            raise ValueError("Argument 'n_agents' must be > 0.")
        if start_date > end_date:
            raise ValueError("Argument 'start_date' must be prior to 'end_date'.")

        tessellation = to_pandas_frame(spatial_tessellation)
        if len(tessellation) < 2:
            raise ValueError("Argument `spatial_tessellation` must contain at least 2 tiles.")
        lats_lngs = tessellation_lat_lngs(tessellation)
        lats = np.ascontiguousarray(lats_lngs[:, 0], dtype=np.float64)
        lngs = np.ascontiguousarray(lats_lngs[:, 1], dtype=np.float64)

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        indipendency_window_s = int(indipendency_window * 3600)
        dt_update_s = int(dt_update_mobSim * 3600)
        seed = int(random_state) if random_state is not None else None

        map_ids = False
        dict_gid_to_uid: dict = {}

        if isinstance(social_graph, str):
            if social_graph != "random":
                raise ValueError("When `social_graph` is a str it must be 'random'.")
            rng_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
            neighbor_starts, neighbors = _core.model_social_graph_random_geometric(n_agents, 0.5, rng_seed)
            neighbor_starts = np.asarray(neighbor_starts, dtype=np.int64)
            neighbors = np.asarray(neighbors, dtype=np.int64)
        elif isinstance(social_graph, list):
            if len(social_graph) == 0:
                raise ValueError("Argument `social_graph` cannot be an empty list.")
            user_ids = sorted({uid for edge in social_graph for uid in edge})
            dict_uid_to_gid = {uid: gid for gid, uid in enumerate(user_ids)}
            dict_gid_to_uid = {gid: uid for uid, gid in dict_uid_to_gid.items()}
            n_agents = len(user_ids)
            map_ids = True
            srcs = np.asarray([dict_uid_to_gid[a] for a, _ in social_graph], dtype=np.int64)
            dsts = np.asarray([dict_uid_to_gid[b] for _, b in social_graph], dtype=np.int64)
            neighbor_starts, neighbors = _core.model_social_graph_edges_to_csr(srcs, dsts, n_agents)
            neighbor_starts = np.asarray(neighbor_starts, dtype=np.int64)
            neighbors = np.asarray(neighbors, dtype=np.int64)
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")

        agent_ids, lats_out, lngs_out, timestamps = _core.model_geosim_simulate_agents(
            lats,
            lngs,
            neighbor_starts,
            neighbors,
            float(self.rho),
            float(self.gamma),
            float(self.alpha),
            float(self.beta),
            float(self.tau),
            float(self.min_wait_time_hours),
            start_ts,
            end_ts,
            indipendency_window_s,
            dt_update_s,
            n_agents,
            seed,
        )

        if map_ids:
            uid_out = [dict_gid_to_uid[int(g) - 1] for g in agent_ids]
        else:
            uid_out = list(agent_ids)

        import datetime

        traj = list(
            zip(
                uid_out,
                lats_out,
                lngs_out,
                [datetime.datetime.fromtimestamp(int(t)) for t in timestamps],
            )
        )
        return trajectory_dataframe(traj)
