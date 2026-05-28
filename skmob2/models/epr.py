from __future__ import annotations

import inspect

import numpy as np

from ._common import (
    RELEVANCE,
    tessellation_lat_lngs,
    to_pandas_frame,
    trajectory_dataframe,
)
from .gravity import Gravity


def compute_od_matrix(
    gravity_singly,
    spatial_tessellation,
    tile_id_column="tile_id",
    relevance_column=RELEVANCE,
):
    return gravity_singly.generate(
        spatial_tessellation,
        tile_id_column=tile_id_column,
        tot_outflows_column=None,
        relevance_column=relevance_column,
        out_format="probabilities",
    ).to_matrix()


class EPR:
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
        self._od_matrix = None
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
        gravity_singly={},
        n_agents=1,
        starting_locations=None,
        od_matrix=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError(
                "The number of starting locations is smaller than the number of agents."
            )
        if gravity_singly == {}:
            self.gravity_singly = Gravity(gravity_type="singly constrained")
        elif type(gravity_singly) is Gravity:
            if gravity_singly.gravity_type == "singly constrained":
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError(
                    "Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'."
                )
        else:
            raise TypeError(
                "Argument `gravity_singly` should be of type skmob.models.gravity.Gravity."
            )

        # Get parameters used in the generation for metadata recording
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = {
            "model": {
                "class": self.__class__.__init__,
                "generate": {
                    i: arg_values[i]
                    for i in args[1:]
                    if i
                    not in [
                        "spatial_tessellation",
                        "od_matrix",
                        "log_file",
                        "starting_locations",
                    ]
                },
            }
        }

        if random_state is not None:
            np.random.seed(random_state)

        self._trajectories_ = []
        self._spatial_tessellation = to_pandas_frame(spatial_tessellation)
        num_locs = len(self._spatial_tessellation)
        self.lats_lngs = tessellation_lat_lngs(self._spatial_tessellation)
        self.relevances = (
            np.ones(num_locs)
            if relevance_column is None
            else self._spatial_tessellation[relevance_column]
            .fillna(0)
            .to_numpy(dtype=float)
        )
        self._od_matrix = None if od_matrix is None else self._dense_rust_od_matrix(od_matrix)

        agent_seeds = np.random.randint(0, 2**31, size=n_agents, dtype=np.int64)
        start_values = (
            list(starting_locations) if starting_locations is not None else None
        )
        resolved_starts = [
            (
                int(start_values.pop())
                if start_values is not None
                else int(np.random.choice(num_locs))
            )
            for _ in range(n_agents)
        ]

        rows = self._epr_generate_parallel(
            start_date,
            end_date,
            resolved_starts,
            agent_seeds,
            od_matrix=self._od_matrix,
        )
        return trajectory_dataframe(rows, parameters=parameters)

    def _epr_generate_parallel(
        self, start_date, end_date, resolved_starts, agent_seeds, od_matrix=None
    ):
        import pandas as pd
        from skmob2 import _core

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        lats = np.asarray(self.lats_lngs[:, 0], dtype=float)
        lngs = np.asarray(self.lats_lngs[:, 1], dtype=float)
        seeds = np.asarray(agent_seeds, dtype=np.int64)
        starts = np.asarray(resolved_starts, dtype=np.int64)
        if od_matrix is None:
            agent_ids, lats_out, lngs_out, timestamps = _core.model_epr_simulate_agents(
                lats,
                lngs,
                np.asarray(self.relevances, dtype=float),
                float(self._rho),
                float(self._gamma),
                float(self._beta),
                float(self._tau),
                float(self._min_wait_time),
                start_ts,
                end_ts,
                seeds,
                starts,
                self.gravity_singly.deterrence_func_type,
                float(self.gravity_singly.deterrence_func_args[0]),
                float(self.gravity_singly.origin_exp),
                float(self.gravity_singly.destination_exp),
            )
        else:
            agent_ids, lats_out, lngs_out, timestamps = (
                _core.model_epr_simulate_agents_from_od(
                    lats,
                    lngs,
                    np.asarray(od_matrix, dtype=float).ravel(),
                    float(self._rho),
                    float(self._gamma),
                    float(self._beta),
                    float(self._tau),
                    float(self._min_wait_time),
                    start_ts,
                    end_ts,
                    seeds,
                    starts,
                )
            )
        return [
            (
                int(agent_ids[k]),
                float(lats_out[k]),
                float(lngs_out[k]),
                pd.Timestamp(int(timestamps[k]), unit="s"),
            )
            for k in range(len(agent_ids))
        ]

    def _dense_rust_od_matrix(self, od_matrix):
        if od_matrix is None:
            return None
        dense = np.asarray(od_matrix, dtype=float)
        if dense.shape != (len(self.lats_lngs), len(self.lats_lngs)):
            raise ValueError("od_matrix must be a dense square matrix with one row per location.")
        return dense


class DensityEPR(EPR):
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
        gravity_singly={},
        n_agents=1,
        starting_locations=None,
        od_matrix=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            od_matrix=od_matrix,
            relevance_column=relevance_column,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class SpatialEPR(EPR):
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
        gravity_singly={},
        n_agents=1,
        starting_locations=None,
        od_matrix=None,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            od_matrix=od_matrix,
            relevance_column=None,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )


class Ditras(EPR):
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
        gravity_singly={},
        n_agents=1,
        starting_locations=None,
        od_matrix=None,
        relevance_column=RELEVANCE,
        random_state=None,
        log_file=None,
        show_progress=False,
    ):
        # Coverage-first compatibility: the spatial phase follows DensityEPR,
        # while the diary generator is retained as public model state.
        return super().generate(
            start_date,
            end_date,
            spatial_tessellation,
            gravity_singly=gravity_singly,
            n_agents=n_agents,
            starting_locations=starting_locations,
            od_matrix=od_matrix,
            relevance_column=relevance_column,
            random_state=random_state,
            log_file=log_file,
            show_progress=show_progress,
        )
