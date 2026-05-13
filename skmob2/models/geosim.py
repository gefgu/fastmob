from __future__ import annotations

import datetime
import logging

import numpy as np

from ._common import require_optional, tessellation_lat_lngs, to_pandas_frame, trajectory_dataframe


class GeoSim:
    def __init__(self, name="GeoSim", rho=0.6, gamma=0.21, alpha=0.2, beta=0.8, tau=17, min_wait_time_hours=1):
        self.name = name
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.min_wait_time_hours = min_wait_time_hours
        self.agents = {}
        self.lats_lngs = []
        self.map_uid_gid = None
        self.dict_uid_to_gid = {}
        self.dict_gid_to_uid = {}

    def uid_2_gid(self, uid):
        return self.dict_uid_to_gid[uid]

    def gid_2_uid(self, gid):
        return self.dict_gid_to_uid[gid]

    def random_weighted_choice(self, weights):
        probabilities = np.asarray(weights, dtype=float) / np.sum(weights)
        return int(np.argmax(np.random.multinomial(1, probabilities)))

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
            }

    def init_social_graph(self, mode="random"):
        igraph = require_optional("igraph")
        Graph = igraph.Graph
        if isinstance(mode, str):
            if mode == "random":
                self.social_graph = Graph.GRG(self.n_agents, 0.5).simplify()
        elif isinstance(mode, list):
            user_ids = sorted(set([uid for edge in mode for uid in edge]))
            self.n_agents = len(user_ids)
            self.dict_uid_to_gid = {uid: gid for gid, uid in enumerate(user_ids)}
            self.dict_gid_to_uid = {gid: uid for uid, gid in self.dict_uid_to_gid.items()}
            self.social_graph = Graph()
            self.social_graph.add_vertices(len(user_ids))
            self.social_graph.add_edges([(self.uid_2_gid(a), self.uid_2_gid(b)) for a, b in mode])

    def assign_starting_location(self, mode="uniform"):
        for i in range(self.n_agents):
            rand_location = int(np.random.randint(0, self.n_locations))
            self.agents[i]["location_vector"][rand_location] = 1
            self.agents[i]["S"] = 1
            self.agents[i]["current_location"] = rand_location
            self.agents[i]["home_location"] = rand_location
            dT = self.get_waiting_time()
            self.agents[i]["time_next_move"] = self.current_date + datetime.timedelta(hours=dT)
            self.agents[i]["dt"] = dT
            out_agent = self.gid_2_uid(i) if self.map_ids else i
            lat, lng = self.lats_lngs[rand_location]
            self.trajectories.append((out_agent, lat, lng, self.current_date))

    def compute_mobility_similarity(self):
        for edge in self.social_graph.es:
            lv1 = self.agents[edge.source]["location_vector"]
            lv2 = self.agents[edge.target]["location_vector"]
            self.social_graph.es(edge.index)["mobility_similarity"] = self.cosine_similarity(lv1, lv2)
            self.social_graph.es(edge.index)["next_update"] = self.current_date + datetime.timedelta(
                hours=self.dt_update_mobSim
            )

    def cosine_similarity(self, x, y):
        den = np.linalg.norm(x) * np.linalg.norm(y)
        return 0.0 if den == 0 else float(np.dot(x, y) / den)

    def get_waiting_time(self):
        powerlaw = require_optional("powerlaw")
        return powerlaw.Truncated_Power_Law(
            xmin=self.min_wait_time_hours, parameters=[1.0 + self.beta, 1.0 / self.tau]
        ).generate_random()[0]

    def store_tmp_movement(self, t, agent, loc, dT):
        self.tmp_upd.append({"agent": agent, "timestamp": t, "location": loc, "dT": dT})

    def update_agent_movement_window(self, to):
        to_remove = []
        for idx, el in enumerate(self.tmp_upd):
            if el["timestamp"] <= to:
                agent = int(el["agent"])
                if self.agents[agent]["location_vector"][el["location"]] == 0:
                    self.agents[agent]["S"] += 1
                self.agents[agent]["location_vector"][el["location"]] += 1
                self.agents[agent]["current_location"] = el["location"]
                out_agent = self.gid_2_uid(agent) if self.map_ids else agent
                lat, lng = self.lats_lngs[el["location"]]
                self.trajectories.append((out_agent, lat, lng, el["timestamp"]))
                to_remove.append(idx)
        for idx in reversed(to_remove):
            self.tmp_upd.pop(idx)

    def make_social_action(self, agent, mode):
        contact_sim = []
        neighbors = list(self.social_graph.neighbors(agent))
        for ns in neighbors:
            eid = self.social_graph.get_eid(agent, ns)
            if self.social_graph.es(eid)["next_update"][0] <= self.current_date:
                lv1 = self.agents[agent]["location_vector"]
                lv2 = self.agents[ns]["location_vector"]
                self.social_graph.es(eid)["mobility_similarity"] = self.cosine_similarity(lv1, lv2)
                self.social_graph.es(eid)["next_update"] = self.current_date + datetime.timedelta(
                    hours=self.dt_update_mobSim
                )
            contact_sim.append(self.social_graph.es(eid)["mobility_similarity"][0])
        if not contact_sim:
            return -1
        contact_pick = (
            self.random_weighted_choice(contact_sim) if np.sum(contact_sim) else np.random.randint(0, len(contact_sim))
        )
        contact = neighbors[int(contact_pick)]
        location_vector_agent = self.agents[agent]["location_vector"]
        location_vector_contact = self.agents[contact]["location_vector"]
        id_locs_feasible = (
            np.where(location_vector_agent == 0)[0]
            if mode == "exploration"
            else np.where(location_vector_agent >= 1)[0]
        )
        if len(id_locs_feasible) == 0:
            return -1
        weights = location_vector_contact[id_locs_feasible]
        if np.sum(weights) == 0:
            return -1
        return int(id_locs_feasible[self.random_weighted_choice(weights)])

    def make_individual_return_action(self, agent):
        v_location = self.agents[agent]["location_vector"]
        id_locs_feasible = np.where(v_location >= 1)[0]
        if len(id_locs_feasible) == 0:
            return -1
        return int(id_locs_feasible[self.random_weighted_choice(v_location[id_locs_feasible])])

    def make_individual_exploration_action(self, agent):
        v_location = self.agents[agent]["location_vector"]
        id_locs_feasible = np.where(v_location == 0)[0]
        if len(id_locs_feasible) == 0:
            return -1
        return int(id_locs_feasible[np.random.randint(0, len(id_locs_feasible))])

    def confirm_action(self, agent, location_id):
        from_ = self.agents[agent]["current_location"]
        self.agents[agent]["current_location"] = location_id
        dT = self.get_waiting_time()
        self.agents[agent]["time_next_move"] = self.current_date + datetime.timedelta(hours=dT)
        self.agents[agent]["dt"] = dT
        self.store_tmp_movement(self.current_date, agent, location_id, dT)
        return {"from": from_, "to": location_id, "next_move": str(self.agents[agent]["time_next_move"])}

    def action_correction(self, agent, choice):
        corrections = []
        if choice == "social_return":
            location_id = self.make_individual_return_action(agent)
            corrections.append("individual_return")
            if location_id < 0:
                choice = "individual_return"
        elif choice == "social_exploration":
            location_id = self.make_individual_exploration_action(agent)
            corrections.append("individual_exploration")
            if location_id < 0:
                choice = "individual_exploration"
        if choice == "individual_return":
            location_id = self.make_individual_exploration_action(agent)
            corrections.append("individual_exploration")
        elif choice == "individual_exploration":
            location_id = self.make_individual_return_action(agent)
            corrections.append("individual_return")
        return location_id, corrections

    def init_spatial_tessellation(self, spatial_tessellation):
        self.spatial_tessellation = to_pandas_frame(spatial_tessellation)
        if len(self.spatial_tessellation) < 2:
            raise ValueError("Argument `spatial_tessellation` must contain at least 2 tiles.")
        self.n_locations = len(self.spatial_tessellation)
        self.lats_lngs = tessellation_lat_lngs(self.spatial_tessellation)

    def init_agents_and_graph(self, social_graph):
        if isinstance(social_graph, str):
            if social_graph != "random":
                raise ValueError("When the argument `social_graph` is a str it must be 'random'.")
            self.map_ids = False
            self.init_agents()
            self.assign_starting_location(mode=self.starting_locations_mode)
            self.init_social_graph(mode=social_graph)
            self.compute_mobility_similarity()
        elif isinstance(social_graph, list):
            if len(social_graph) == 0:
                raise ValueError("The argument `social_graph` cannot be an empty list.")
            self.map_ids = True
            self.init_social_graph(mode=social_graph)
            self.init_agents()
            self.assign_starting_location(mode=self.starting_locations_mode)
            self.compute_mobility_similarity()
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")

    def _choose_location(self, agent):
        p_exp = self.agents[agent]["rho"] * (self.agents[agent]["S"] ** -self.agents[agent]["gamma"])
        p_rand_exp = np.random.rand()
        p_rand_soc = np.random.rand()
        if p_rand_exp < p_exp:
            choice = "social_exploration" if p_rand_soc < self.agents[agent]["alpha"] else "individual_exploration"
        else:
            choice = "social_return" if p_rand_soc < self.agents[agent]["alpha"] else "individual_return"
        if choice == "social_exploration":
            location_id = self.make_social_action(agent, "exploration")
        elif choice == "individual_exploration":
            location_id = self.make_individual_exploration_action(agent)
        elif choice == "social_return":
            location_id = self.make_social_action(agent, "return")
        else:
            location_id = self.make_individual_return_action(agent)
        corrections = None
        if location_id == -1:
            location_id, corrections = self.action_correction(agent, choice)
        return choice, location_id, corrections

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
        if random_state is not None:
            np.random.seed(random_state)
        if log_file is not None:
            logging.basicConfig(format="%(message)s", filename=log_file, filemode="w", level=logging.INFO)
        self.n_agents = n_agents
        self.tmp_upd = []
        self.trajectories = []
        self.dt_update_mobSim = dt_update_mobSim
        self.indipendency_window = indipendency_window
        self.verbose = verbose
        self.log_file = log_file
        self.starting_locations_mode = "uniform"
        self.start_date, self.current_date, self.end_date = start_date, start_date, end_date
        delta_T = self.end_date - self.start_date
        self.total_h = delta_T.days * 24 + delta_T.seconds // 3600
        self.init_spatial_tessellation(spatial_tessellation)
        self.init_agents_and_graph(social_graph)

        while self.current_date < self.end_date:
            self.update_agent_movement_window(self.current_date - datetime.timedelta(hours=self.indipendency_window))
            min_time_next_move = self.end_date
            for agent in range(self.n_agents):
                if self.current_date != self.agents[agent]["time_next_move"]:
                    if self.agents[agent]["time_next_move"] < min_time_next_move:
                        min_time_next_move = self.agents[agent]["time_next_move"]
                    continue
                _choice, location_id, _corrections = self._choose_location(agent)
                if location_id < 0:
                    raise Exception("Fatal error, unable to correct the location")
                self.confirm_action(agent, location_id)
                if self.agents[agent]["time_next_move"] < min_time_next_move:
                    min_time_next_move = self.agents[agent]["time_next_move"]
            self.current_date = min_time_next_move
        if log_file is not None:
            logging.shutdown()
        self.update_agent_movement_window(self.end_date)
        return trajectory_dataframe(self.trajectories)
