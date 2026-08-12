from __future__ import annotations

import secrets

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob import _core
from fastmob.core import Locations, TrajDataFrame

from .markov_diary_generator import MarkovDiaryGenerator


class STS_epr:
    """STS-EPR (Spatial, Temporal, and Social EPR) trajectory generator.

    STS-EPR extends :class:`GeoSim` by incorporating a temporal diary (via
    :class:`MarkovDiaryGenerator`) and a gravity-weighted individual exploration
    mechanism [CRP2020]_. Each agent follows a pre-generated mobility diary that
    dictates *when* to move (home vs. away), while the social and EPR mechanisms
    determine *where* to go.

    **Action selection.** With probability
    :math:`P_{\\text{exp}} = \\rho S^{-\\gamma}` the agent explores a new location;
    otherwise it returns. With probability :math:`\\alpha` a social contact influences
    the destination (Social); with probability :math:`1 - \\alpha` the agent acts
    alone (Individual). Diary entries with ``abstract_location == 0`` always force a
    return to the home location.

    **Individual Exploration.** An unvisited location :math:`j \\neq i` is selected
    with probability proportional to the gravity score:

    .. math::

        p_{ij} = \\frac{r_i r_j}{d_{ij}^2}

    where :math:`r_{i(j)}` is location relevance and :math:`d_{ij}` is the Haversine
    distance.

    **Social Exploration.** A social contact :math:`c` is selected with probability
    proportional to cosine mobility similarity. A location from
    :math:`\\text{exp}(a) \\cap \\text{ret}(c)` is chosen proportionally to
    :math:`c`'s visitation frequency.

    **Individual Return.** A previously visited location :math:`j \\in \\text{ret}(a)`
    is chosen with probability :math:`\\Pi_j = f_j`.

    **Social Return.** As in Social Exploration, but from
    :math:`\\text{ret}(a) \\cap \\text{ret}(c)`.

    Parameters
    ----------
    name : str, optional
        Human-readable label. The default is ``"STS-EPR"``.
    rho : float, optional
        Exploration parameter :math:`\\rho \\in (0, 1]`. The default is ``0.6``
        [SKWB2010]_.
    gamma : float, optional
        Return-penalty exponent :math:`\\gamma \\geq 0`. The default is ``0.21``
        [SKWB2010]_.
    alpha : float, optional
        Social influence probability :math:`\\alpha \\in [0, 1]`. The default is
        ``0.2`` [THSG2015]_.

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

    Notes
    -----
    ``spatial_tessellation`` must contain at least 3 tiles and a relevance column.
    The :class:`MarkovDiaryGenerator` must be fitted before being passed to
    :meth:`generate`.

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
    .. [CRP2020] Cornacchia, G., Rossetti, G. & Pappalardo, L. (2020). Modelling
       Human Mobility considering Spatial, Temporal and Social Dimensions.
    .. [PS2018] Pappalardo, L. & Simini, F. (2018). Data-driven generation of
       spatio-temporal routines in human mobility. *Data Mining and Knowledge
       Discovery*, 32, 787–829.

    See Also
    --------
    GeoSim, Ditras, MarkovDiaryGenerator
    """

    def __init__(self, name="STS-EPR", rho=0.6, gamma=0.21, alpha=0.2):
        self.name = name
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha

    def generate(
        self,
        start_date,
        end_date,
        spatial_tessellation,
        diary_generator,
        social_graph="random",
        n_agents=500,
        rsl=False,
        distance_matrix=None,
        relevance_column=None,
        min_relevance=0.1,
        dt_update_mobSim=24 * 7,
        indipendency_window=0.5,
        random_state=None,
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
            ``geometry`` column or explicit ``lat`` / ``lng`` columns, and a
            relevance column. Must contain at least 3 tiles.
        diary_generator : MarkovDiaryGenerator
            A fitted :class:`MarkovDiaryGenerator` that supplies the temporal
            mobility diary for each agent.
        social_graph : ``"random"`` or list of (uid, uid) tuples, optional
            Social network for the agents.

            - ``"random"`` — a random geometric graph is generated automatically.
            - A list of ``(uid_a, uid_b)`` edge tuples — the number of agents is
              inferred from the unique node set.

            The default is ``"random"``.
        n_agents : int, optional
            Number of agents when ``social_graph="random"``. The default is ``500``.
        rsl : bool, optional
            If ``True``, the starting location for each agent is sampled with
            probability proportional to the location's relevance. If ``False``,
            it is chosen uniformly at random. The default is ``False``.
        distance_matrix : numpy.ndarray or None, optional
            Pre-computed ``(n_locs, n_locs)`` distance matrix in kilometres. If
            ``None``, the Rust simulator uses cached gravity OD rows instead of a
            full matrix. The default is ``None``.
        relevance_column : str or None, optional
            Name of the column in ``spatial_tessellation`` used as location
            relevance. The default is ``None`` (falls back to ``"relevance"``).
        min_relevance : float, optional
            Value substituted for any zero-relevance location to avoid division by
            zero. The default is ``0.1``.
        dt_update_mobSim : float, optional
            Interval in hours between social-graph mobility-similarity updates.
            The default is ``168`` (one week).
        indipendency_window : float, optional
            Time window in hours before an agent's move can influence others.
            The default is ``0.5``.
        random_state : int or None, optional
            Random seed for reproducibility. The default is ``None``.
        Returns
        -------
        DataFrame
            Synthetic trajectories with columns ``uid``, ``datetime``, ``lat``,
            and ``lng``.

        Raises
        ------
        TypeError
            If ``diary_generator`` is not a :class:`MarkovDiaryGenerator` or
            ``rsl`` is not a bool.
        ValueError
            If ``n_agents <= 0``, ``start_date > end_date``, ``social_graph`` is
            an empty list, or the social graph string is not ``"random"``.
        IndexError
            If ``relevance_column`` is not found in ``spatial_tessellation``.
        """
        if not isinstance(diary_generator, MarkovDiaryGenerator):
            raise TypeError("Argument `diary_generator` should be of type MarkovDiaryGenerator.")
        if type(rsl) is not bool:
            raise TypeError("Argument `rsl` must be a bool.")
        if n_agents <= 0:
            raise ValueError("Argument 'n_agents' must be > 0.")
        if start_date > end_date:
            raise ValueError("Argument 'start_date' must be prior to 'end_date'.")
        if not hasattr(diary_generator, "_cdf_matrix_flat"):
            raise ValueError("diary_generator has not been fitted. Call fit() first.")

        if not isinstance(spatial_tessellation, Locations):
            raise TypeError(
                "Spatial generation models require Locations(scope='global'); use Locations.from_tessellation(...) for legacy inputs"
            )
        prepared = spatial_tessellation.model_input()
        tessellation = prepared.frame
        if len(tessellation) < 3:
            raise ValueError("Argument `spatial_tessellation` must contain at least 3 tiles.")
        lats = prepared.latitudes
        lngs = prepared.longitudes

        col = relevance_column if relevance_column is not None else "relevance"
        if col not in tessellation.columns:
            raise IndexError(f"Column '{col}' not found in spatial_tessellation.")
        relevances = pc.if_else(
            pc.equal(prepared.values(col), 0.0),
            pa.scalar(float(min_relevance)),
            prepared.values(col),
        )

        if distance_matrix is not None:
            flat_distances = pa.array([distance for row in distance_matrix for distance in row], type=pa.float64())
        else:
            flat_distances = pa.array([], type=pa.float64())

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        total_h = (end_ts - start_ts) // 3600
        indipendency_window_s = int(indipendency_window * 3600)
        dt_update_s = int(dt_update_mobSim * 3600)
        seed = int(random_state) if random_state is not None else None

        map_ids = False
        dict_gid_to_uid: dict = {}

        if isinstance(social_graph, str):
            if social_graph != "random":
                raise ValueError("When `social_graph` is a str it must be 'random'.")
            rng_seed = seed if seed is not None else secrets.randbits(31)
            neighbor_starts, neighbors = _core.model_social_graph_random_geometric_arrow(n_agents, 0.5, rng_seed)
        elif isinstance(social_graph, list):
            if len(social_graph) == 0:
                raise ValueError("Argument `social_graph` cannot be an empty list.")
            user_ids = sorted({uid for edge in social_graph for uid in edge})
            dict_uid_to_gid = {uid: gid for gid, uid in enumerate(user_ids)}
            dict_gid_to_uid = {gid: uid for uid, gid in dict_uid_to_gid.items()}
            n_agents = len(user_ids)
            map_ids = True
            srcs = pa.array([dict_uid_to_gid[a] for a, _ in social_graph], type=pa.int64())
            dsts = pa.array([dict_uid_to_gid[b] for _, b in social_graph], type=pa.int64())
            neighbor_starts, neighbors = _core.model_social_graph_edges_to_csr_arrow(srcs, dsts, n_agents)
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")

        diary_seed = seed if seed is not None else secrets.randbits(31)
        flat_ts, flat_locs, d_starts, d_ends = _core.markov_diary_batch_generate_arrow(
            diary_generator._cdf_matrix_flat,
            total_h,
            start_ts,
            n_agents,
            diary_seed,
        )
        agent_ids, lats_out, lngs_out, timestamps = _core.model_sts_epr_simulate_agents_arrow(
            lats,
            lngs,
            relevances,
            flat_distances,
            neighbor_starts,
            neighbors,
            flat_ts,
            flat_locs,
            pa.array(d_starts, type=pa.int64()),
            pa.array(d_ends, type=pa.int64()),
            float(self.rho),
            float(self.gamma),
            float(self.alpha),
            start_ts,
            end_ts,
            indipendency_window_s,
            dt_update_s,
            n_agents,
            seed,
            None,
            rsl,
        )

        agent_values = agent_ids.to_pylist()
        if map_ids:
            uid_out = [dict_gid_to_uid[int(g) - 1] for g in agent_values]
        else:
            uid_out = agent_values
        return TrajDataFrame(
            nw.from_dict(
                {
                    "uid": uid_out,
                    "lat": pa.array(lats_out),
                    "lng": pa.array(lngs_out),
                    "datetime": pc.cast(pa.array(timestamps), pa.timestamp("s")),
                },
                backend=tessellation.implementation,
            )
            .sort(["uid", "datetime"])
            .select(["uid", "datetime", "lat", "lng"])
            .to_native()
        )
