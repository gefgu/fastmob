from __future__ import annotations

import datetime
import math

import numpy as np

from ._common import EARTH_RADIUS_KM, haversine_km, tessellation_lat_lngs, to_pandas_frame
from .geosim import GeoSim
from .markov_diary_generator import MarkovDiaryGenerator


class STS_epr(GeoSim):
    def __init__(self, name="STS-EPR", rho=0.6, gamma=0.21, alpha=0.2):
        super().__init__(name=name, rho=rho, gamma=gamma, alpha=alpha, beta=0.8, tau=17, min_wait_time_hours=1)
        self.distance_matrix = None

    def assign_starting_location(self, mode="uniform"):
        for i in range(self.n_agents):
            if mode == "relevance":
                p_location = self.relevances / np.sum(self.relevances)
                rand_location = int(np.where(np.random.multinomial(1, p_location) == 1)[0][0])
            else:
                rand_location = int(np.random.randint(0, self.n_locations))
            self.agents[i]["location_vector"][rand_location] = 1
            self.agents[i]["S"] = 1
            self.agents[i]["current_location"] = rand_location
            self.agents[i]["home_location"] = rand_location
            self.agents[i]["time_next_move"] = self.agents[i]["mobility_diary"].loc[1]["datetime"]
            self.agents[i]["index_mobility_diary"] = 1
            self.agents[i]["dt"] = 1
            out_agent = self.gid_2_uid(i) if self.map_ids else i
            lat, lng = self.lats_lngs[rand_location]
            self.trajectories.append((out_agent, lat, lng, self.current_date))

    def init_agents(self):
        self.agents = {}
        for i in range(self.n_agents):
            self.agents[i] = {
                "ID": i,
                "current_location": -1,
                "home_location": -1,
                "location_vector": np.array([0] * self.n_locations),
                "S": 0,
                "alpha": self.alpha,
                "rho": self.rho,
                "gamma": self.gamma,
                "time_next_move": self.start_date,
                "dt": 0,
                "mobility_diary": None,
                "index_mobility_diary": None,
            }

    def compute_distance_matrix(self):
        self.distance_matrix = np.zeros((len(self.spatial_tessellation), len(self.spatial_tessellation)))
        for i in range(len(self.spatial_tessellation)):
            for j in range(len(self.spatial_tessellation)):
                if i != j:
                    self.distance_matrix[i, j] = self.distance_earth_km(
                        {"lat": self.lats_lngs[i][0], "lon": self.lats_lngs[i][1]},
                        {"lat": self.lats_lngs[j][0], "lon": self.lats_lngs[j][1]},
                    )

    def compute_od_row(self, row):
        if self.distance_matrix[row, 0] != 0 or self.distance_matrix[row, 1] != 0:
            return
        lat1_r = math.radians(float(self.lats_lngs[row, 0]))
        lng1_r = math.radians(float(self.lats_lngs[row, 1]))
        lats_r = np.radians(self.lats_lngs[:, 0])
        lngs_r = np.radians(self.lats_lngs[:, 1])
        dlat = lat1_r - lats_r
        dlng = lng1_r - lngs_r
        a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1_r) * np.cos(lats_r) * np.sin(dlng / 2.0) ** 2
        distances = EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))
        distances[row] = 0.0
        self.distance_matrix[row] = distances

    def distance_earth_km(self, src, dest):
        return haversine_km((src["lat"], src["lon"]), (dest["lat"], dest["lon"]))

    def init_mobility_diaries(self, hours, start_date):
        import pandas as pd

        for i in range(self.n_agents):
            rand_seed_diary = np.random.randint(0, 10**6)
            short_diary = self.diary_generator._generate_list(hours, start_date, random_state=rand_seed_diary)
            attempts = 0
            while len(short_diary) < 2 and attempts < 100:
                rand_seed_diary = np.random.randint(0, 10**6)
                short_diary = self.diary_generator._generate_list(hours, start_date, random_state=rand_seed_diary)
                attempts += 1
            if len(short_diary) < 2:
                short_diary.append([start_date + datetime.timedelta(hours=1), 0])
            self.agents[i]["mobility_diary"] = pd.DataFrame(short_diary, columns=["datetime", "abstract_location"])

    def get_current_abstract_location_from_diary(self, agent):
        row = self.agents[agent]["index_mobility_diary"]
        return self.agents[agent]["mobility_diary"].loc[row]["abstract_location"]

    def confirm_action(self, agent, location_id):
        from_ = self.agents[agent]["current_location"]
        self.agents[agent]["current_location"] = location_id
        self.agents[agent]["index_mobility_diary"] += 1
        row_diary = self.agents[agent]["index_mobility_diary"]
        if row_diary < len(self.agents[agent]["mobility_diary"]):
            self.agents[agent]["time_next_move"] = self.agents[agent]["mobility_diary"].loc[row_diary]["datetime"]
            delta_T = self.agents[agent]["time_next_move"] - self.current_date
            dT = delta_T.days * 24 + delta_T.seconds // 3600
            next_move = str(self.agents[agent]["time_next_move"])
        else:
            self.agents[agent]["time_next_move"] = self.end_date + datetime.timedelta(hours=1)
            dT = 1
            next_move = "None"
        self.agents[agent]["dt"] = dT
        self.store_tmp_movement(self.current_date, agent, location_id, dT)
        return {"from": from_, "to": location_id, "next_move": next_move}

    def action_correction_diary(self, agent, choice):
        return self.action_correction(agent, choice)

    def init_spatial_tessellation(self, spatial_tessellation, relevance_column, min_relevance):
        self.spatial_tessellation = to_pandas_frame(spatial_tessellation)
        if len(self.spatial_tessellation) < 3:
            raise ValueError("Argument `spatial_tessellation` must contain at least 3 tiles.")
        self.n_locations = len(self.spatial_tessellation)
        self.lats_lngs = tessellation_lat_lngs(self.spatial_tessellation)
        if relevance_column not in self.spatial_tessellation.columns:
            raise IndexError("the column `relevance_columns` is invalid")
        self.relevances = np.asarray(self.spatial_tessellation[relevance_column], dtype=float)
        self.relevances = np.where(self.relevances == 0, min_relevance, self.relevances)

    def init_agents_and_graph(self, social_graph):
        if isinstance(social_graph, str):
            if social_graph != "random":
                raise ValueError("When the argument `social_graph` is a str it must be 'random'.")
            self.map_ids = False
            self.init_agents()
            self.init_mobility_diaries(self.total_h, self.start_date)
            self.assign_starting_location(mode=self.starting_locations_mode)
            self.init_social_graph(mode=social_graph)
            self.compute_mobility_similarity()
        elif isinstance(social_graph, list):
            if len(social_graph) == 0:
                raise ValueError("The argument `social_graph` cannot be an empty list.")
            self.map_ids = True
            self.init_social_graph(mode=social_graph)
            self.init_agents()
            self.init_mobility_diaries(self.total_h, self.start_date)
            self.assign_starting_location(mode=self.starting_locations_mode)
            self.compute_mobility_similarity()
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")

    def make_individual_exploration_action(self, agent):
        v_location = self.agents[agent]["location_vector"]
        id_locs_feasible = np.where(v_location == 0)[0]
        constrained = {self.agents[agent]["current_location"], self.agents[agent]["home_location"]}
        id_locs_feasible = [loc for loc in id_locs_feasible if loc not in constrained]
        if len(id_locs_feasible) == 0:
            return -1
        src = self.agents[agent]["current_location"]
        self.compute_od_row(src)
        distance_row = np.asarray(self.distance_matrix[src], dtype=float)
        distance_row[src] = 1.0
        score = (1.0 / distance_row**2) * self.relevances * self.relevances[src]
        weights = np.asarray([score[i] for i in id_locs_feasible], dtype=float)
        if np.sum(weights) == 0:
            return int(id_locs_feasible[np.random.randint(0, len(id_locs_feasible))])
        return int(id_locs_feasible[self.random_weighted_choice(weights)])

    def _choose_location(self, agent):
        abstract_location = self.get_current_abstract_location_from_diary(agent)
        if abstract_location == 0:
            return "home_return", self.agents[agent]["home_location"], None
        return super()._choose_location(agent)

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
                "Argument `diary_generator` should be of type skmob.models.markov_diary_generator.MarkovDiaryGenerator."
            )
        if type(rsl) is not bool:
            raise TypeError("Argument `rsl` must be a bool.")
        self.diary_generator = diary_generator
        self.starting_locations_mode = "relevance" if rsl else "uniform"
        if n_agents <= 0:
            raise ValueError("Argument 'n_agents' must be > 0.")
        if start_date > end_date:
            raise ValueError("Argument 'start_date' must be prior to 'end_date'.")
        if random_state is not None:
            np.random.seed(random_state)
        self.n_agents = n_agents
        self.tmp_upd = []
        self.trajectories = []
        self.dt_update_mobSim = dt_update_mobSim
        self.indipendency_window = indipendency_window
        self.verbose = verbose
        self.log_file = log_file
        self.start_date, self.current_date, self.end_date = start_date, start_date, end_date
        delta_T = self.end_date - self.start_date
        self.total_h = delta_T.days * 24 + delta_T.seconds // 3600
        self.init_spatial_tessellation(spatial_tessellation, relevance_column, min_relevance)
        self.distance_matrix = (
            np.asarray(distance_matrix, dtype=float)
            if distance_matrix is not None
            else np.zeros((len(self.spatial_tessellation), len(self.spatial_tessellation)))
        )
        self.init_agents_and_graph(social_graph)
        while self.current_date < self.end_date:
            self.update_agent_movement_window(self.current_date - datetime.timedelta(hours=self.indipendency_window))
            min_time_next_move = self.end_date
            for agent in range(self.n_agents):
                if self.current_date != self.agents[agent]["time_next_move"]:
                    if self.agents[agent]["time_next_move"] < min_time_next_move:
                        min_time_next_move = self.agents[agent]["time_next_move"]
                    continue
                choice, location_id, corrections = self._choose_location(agent)
                if location_id == -1:
                    location_id, corrections = self.action_correction_diary(agent, choice)
                if location_id < 0:
                    raise Exception("Fatal error, unable to correct the location")
                self.confirm_action(agent, location_id)
                if self.agents[agent]["time_next_move"] < min_time_next_move:
                    min_time_next_move = self.agents[agent]["time_next_move"]
            self.current_date = min_time_next_move
        self.update_agent_movement_window(self.end_date)
        from ._common import trajectory_dataframe

        return trajectory_dataframe(self.trajectories)
