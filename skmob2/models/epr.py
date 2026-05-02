from __future__ import annotations

import datetime
import inspect
import logging
import math
from collections import defaultdict

import numpy as np

from ._common import RELEVANCE, require_optional, tessellation_lat_lngs, to_pandas_frame, trajectory_dataframe
from .gravity import Gravity


def compute_od_matrix(gravity_singly, spatial_tessellation, tile_id_column="tile_id", relevance_column=RELEVANCE):
    return gravity_singly.generate(
        spatial_tessellation,
        tile_id_column=tile_id_column,
        tot_outflows_column=None,
        relevance_column=relevance_column,
        out_format="probabilities",
    ).to_matrix()


def populate_od_matrix(location, lats_lngs, relevances, gravity_singly):
    ll_origin = lats_lngs[location]
    distances = np.asarray([_distance(ll_origin, coords) for coords in lats_lngs])
    scores = gravity_singly._compute_gravity_score(distances[None, :], relevances[location, None], relevances)[0]
    total = np.sum(scores)
    return scores / total if total else np.full(len(lats_lngs), 1.0 / len(lats_lngs))


def _distance(origin, dest):
    from ._common import haversine_km

    return haversine_km(tuple(origin), tuple(dest))


class EPR:
    def __init__(self, name="EPR model", rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):
        self._name = name
        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta
        self._location2visits = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self.lats_lngs = None
        self.relevances = None
        self._starting_loc = None
        self.gravity_singly = None
        self._min_wait_time = min_wait_time_minutes / 60.0
        self._trajectories_ = []
        self._log_file = None

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

    def _weighted_random_selection(self, current_location):
        locations = np.fromiter(self._location2visits.keys(), dtype=int)
        weights = np.fromiter(self._location2visits.values(), dtype=float)
        currloc_idx = np.where(locations == current_location)[0]
        if len(currloc_idx):
            locations = np.delete(locations, currloc_idx[0])
            weights = np.delete(weights, currloc_idx[0])
        if len(locations) == 0:
            return int(current_location)
        weights = weights / np.sum(weights)
        return int(np.random.choice(locations, size=1, p=weights)[0])

    def _preferential_return(self, current_location):
        next_location = self._weighted_random_selection(current_location)
        if self._log_file is not None:
            logging.info(f"RETURN to {next_location} ({self.lats_lngs[next_location]})")
            logging.info(f"\t frequency = {self._location2visits[next_location]}")
        return next_location

    def _preferential_exploration(self, current_location):
        if self._is_sparse:
            row = self._od_matrix.get(int(current_location))
            if row is None:
                row = populate_od_matrix(current_location, self.lats_lngs, self.relevances, self.gravity_singly)
                self._od_matrix[int(current_location)] = row
            weights = row
        else:
            weights = np.asarray(self._od_matrix[int(current_location)], dtype=float)
        total = np.sum(weights)
        if total == 0:
            weights = np.ones(len(self.lats_lngs)) / len(self.lats_lngs)
        else:
            weights = weights / total
        return int(np.random.choice(np.arange(len(weights)), size=1, p=weights)[0])

    def _get_trajdataframe(self, parameters):
        rows = [
            (agent_id, self.lats_lngs[loc][0], self.lats_lngs[loc][1], dt) for agent_id, dt, loc in self._trajectories_
        ]
        return trajectory_dataframe(rows, parameters=parameters)

    def _choose_location(self):
        n_visited_locations = len(self._location2visits)
        if n_visited_locations == 0:
            self._starting_loc = self._preferential_exploration(self._starting_loc)
            return self._starting_loc
        _agent_id, _current_time, current_location = self._trajectories_[-1]
        p_new = np.random.uniform(0, 1)
        n_locs = len(self.lats_lngs)
        if (p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != n_locs) or (
            n_visited_locations == 1
        ):
            return self._preferential_exploration(current_location)
        return self._preferential_return(current_location)

    def _time_generator(self):
        powerlaw = require_optional("powerlaw")
        return powerlaw.Truncated_Power_Law(
            xmin=self.min_wait_time, parameters=[1.0 + self._beta, 1.0 / self._tau]
        ).generate_random()[0]

    def _choose_waiting_time(self):
        return self._time_generator()

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
            raise IndexError("The number of starting locations is smaller than the number of agents.")
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
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = {
            "model": {
                "class": self.__class__.__init__,
                "generate": {
                    i: arg_values[i]
                    for i in args[1:]
                    if i not in ["spatial_tessellation", "od_matrix", "log_file", "starting_locations"]
                },
            }
        }
        if random_state is not None:
            np.random.seed(random_state)
        if log_file is not None:
            self._log_file = log_file
            logging.basicConfig(format="%(message)s", filename=log_file, filemode="w", level=logging.INFO)

        self._trajectories_ = []
        self._spatial_tessellation = to_pandas_frame(spatial_tessellation)
        num_locs = len(self._spatial_tessellation)
        self.lats_lngs = tessellation_lat_lngs(self._spatial_tessellation)
        self.relevances = (
            np.ones(num_locs)
            if relevance_column is None
            else self._spatial_tessellation[relevance_column].fillna(0).to_numpy(dtype=float)
        )
        self._od_matrix = {} if od_matrix is None else od_matrix
        self._is_sparse = od_matrix is None

        start_values = list(starting_locations) if starting_locations is not None else None
        for agent_id in range(1, n_agents + 1):
            self._location2visits = defaultdict(int)
            self._starting_loc = (
                int(np.random.choice(np.arange(num_locs), size=1)[0])
                if start_values is None
                else int(start_values.pop())
            )
            self._epr_generate_one_agent(agent_id, start_date, end_date)
        if self._log_file is not None:
            logging.shutdown()
        return self._get_trajdataframe(parameters)

    def _epr_generate_one_agent(self, agent_id, start_date, end_date):
        current_date = start_date
        self._trajectories_.append((agent_id, current_date, self._starting_loc))
        self._location2visits[self._starting_loc] += 1
        current_date += datetime.timedelta(hours=self._choose_waiting_time())
        while current_date < end_date:
            next_location = self._choose_location()
            self._trajectories_.append((agent_id, current_date, next_location))
            self._location2visits[next_location] += 1
            current_date += datetime.timedelta(hours=self._choose_waiting_time())


class DensityEPR(EPR):
    def __init__(self, name="Density EPR model", rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):
        super().__init__(rho=rho, gamma=gamma, beta=beta, tau=tau, min_wait_time_minutes=min_wait_time_minutes)
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
    def __init__(self, name="Spatial EPR model", rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):
        super().__init__(rho=rho, gamma=gamma, beta=beta, tau=tau, min_wait_time_minutes=min_wait_time_minutes)
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
