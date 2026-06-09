from __future__ import annotations

import numpy as np

from skmob2 import _core

from ._common import tessellation_lat_lngs, to_pandas_frame, trajectory_dataframe
from .markov_diary_generator import MarkovDiaryGenerator


class STS_epr:
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
        log_file=None,
        verbose=0,
        show_progress=False,
    ):
        if not isinstance(diary_generator, MarkovDiaryGenerator):
            raise TypeError(
                "Argument `diary_generator` should be of type MarkovDiaryGenerator."
            )
        if type(rsl) is not bool:
            raise TypeError("Argument `rsl` must be a bool.")
        if n_agents <= 0:
            raise ValueError("Argument 'n_agents' must be > 0.")
        if start_date > end_date:
            raise ValueError("Argument 'start_date' must be prior to 'end_date'.")
        if not hasattr(diary_generator, "_cdf_matrix_flat"):
            raise ValueError("diary_generator has not been fitted. Call fit() first.")

        tessellation = to_pandas_frame(spatial_tessellation)
        if len(tessellation) < 3:
            raise ValueError("Argument `spatial_tessellation` must contain at least 3 tiles.")
        lats_lngs = tessellation_lat_lngs(tessellation)
        lats = np.ascontiguousarray(lats_lngs[:, 0], dtype=np.float64)
        lngs = np.ascontiguousarray(lats_lngs[:, 1], dtype=np.float64)

        col = relevance_column if relevance_column is not None else "relevance"
        if col not in tessellation.columns:
            raise IndexError(f"Column '{col}' not found in spatial_tessellation.")
        relevances = np.asarray(tessellation[col], dtype=np.float64)
        relevances = np.where(relevances == 0, min_relevance, relevances)
        relevances = np.ascontiguousarray(relevances)

        if distance_matrix is not None:
            flat_distances = np.ascontiguousarray(
                np.asarray(distance_matrix, dtype=np.float64).ravel()
            )
        else:
            flat_distances = np.ascontiguousarray(
                _core.model_distance_matrix_numpy(lats, lngs), dtype=np.float64
            )

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
            neighbor_starts, neighbors = _core.model_social_graph_edges_to_csr(
                srcs, dsts, n_agents
            )
            neighbor_starts = np.asarray(neighbor_starts, dtype=np.int64)
            neighbors = np.asarray(neighbors, dtype=np.int64)
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")

        diary_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        flat_ts, flat_locs, d_starts, d_ends = _core.markov_diary_batch_generate(
            diary_generator._cdf_matrix_flat,
            total_h,
            start_ts,
            n_agents,
            diary_seed,
        )
        diary_timestamps = np.asarray(flat_ts, dtype=np.int64)
        diary_abs_locs = np.asarray(flat_locs, dtype=np.int32)
        diary_starts = np.asarray(d_starts, dtype=np.int64)
        diary_ends = np.asarray(d_ends, dtype=np.int64)

        agent_ids, lats_out, lngs_out, timestamps = _core.model_sts_epr_simulate_agents(
            lats,
            lngs,
            relevances,
            flat_distances,
            neighbor_starts,
            neighbors,
            diary_timestamps,
            diary_abs_locs,
            diary_starts,
            diary_ends,
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
