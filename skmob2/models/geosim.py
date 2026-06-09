from __future__ import annotations

import numpy as np

from skmob2 import _core

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
            neighbor_starts, neighbors = _core.model_social_graph_random_geometric(
                n_agents, 0.5, rng_seed
            )
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
