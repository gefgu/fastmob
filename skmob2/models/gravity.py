from __future__ import annotations

from typing import Any

import numpy as np

from skmob2 import _core

from ._common import (
    FLOW,
    RELEVANCE,
    TILE_ID,
    TOT_OUTFLOW,
    flow_dataframe,
    haversine_km,
    tessellation_lat_lngs,
    to_pandas_frame,
)


def ci(i, number_locs):
    c = list(np.zeros(number_locs))
    c[i] = 1.0
    return c


def exponential_deterrence_func(x, R):
    return np.exp(-x * R)


def powerlaw_deterrence_func(x, exponent):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, exponent)


def compute_distance_matrix(spatial_tessellation: Any, origins):
    coords = tessellation_lat_lngs(spatial_tessellation)
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=float)
    for id_i in origins:
        for id_j in range(int(id_i) + 1, n):
            distance = haversine_km(tuple(coords[int(id_i)]), tuple(coords[id_j]))
            distance_matrix[int(id_i), id_j] = distance
            distance_matrix[id_j, int(id_i)] = distance
    return distance_matrix


class Gravity:
    """Gravity model compatible with original scikit-mobility generation APIs."""

    def __init__(
        self,
        deterrence_func_type="power_law",
        deterrence_func_args=[-2.0],
        origin_exp=1.0,
        destination_exp=1.0,
        gravity_type="singly constrained",
        name="Gravity model",
    ):
        self._name = name
        self._deterrence_func_args = deterrence_func_args
        self._origin_exp = origin_exp
        self._destination_exp = destination_exp
        self._gravity_type = gravity_type
        if deterrence_func_type not in ("power_law", "exponential"):
            print(
                'Deterrence function type "%s" not available. Power law will be used.\n'
                "Available deterrence functions are [power_law, exponential]" % deterrence_func_type
            )
            deterrence_func_type = "power_law"
        self._deterrence_func_type = deterrence_func_type

    @property
    def name(self):
        return self._name

    @property
    def deterrence_func_type(self):
        return self._deterrence_func_type

    @property
    def deterrence_func_args(self):
        return self._deterrence_func_args

    @property
    def origin_exp(self):
        return self._origin_exp

    @property
    def destination_exp(self):
        return self._destination_exp

    @property
    def gravity_type(self):
        return self._gravity_type

    def __str__(self):
        return (
            'Gravity(name="%s", deterrence_func_type="%s", deterrence_func_args=%s, '
            'origin_exp=%s, destination_exp=%s, gravity_type="%s")'
            % (
                self._name,
                self._deterrence_func_type,
                self._deterrence_func_args,
                self._origin_exp,
                self._destination_exp,
                self._gravity_type,
            )
        )

    def generate(
        self,
        spatial_tessellation,
        tile_id_column=TILE_ID,
        tot_outflows_column=TOT_OUTFLOW,
        relevance_column=RELEVANCE,
        out_format="flows",
    ):
        spatial_tessellation = to_pandas_frame(spatial_tessellation)
        n_locs = len(spatial_tessellation)
        relevances = spatial_tessellation[relevance_column].fillna(0).to_numpy(dtype=float)
        self._tile_id_column = tile_id_column

        if out_format not in ["flows", "flows_sample", "probabilities"]:
            print(
                'Output format "%s" not available. Flows will be used.\n'
                "Available output formats are [flows, flows_sample, probabilities]" % out_format
            )
            out_format = "flows"

        if "flows" in out_format:
            if tot_outflows_column not in spatial_tessellation.columns:
                raise KeyError("The column 'tot_outflows' must be present in the tessellation.")
            tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).to_numpy(dtype=float)
        else:
            tot_outflows = np.zeros(n_locs, dtype=float)

        origins = np.arange(n_locs)
        coords = tessellation_lat_lngs(spatial_tessellation)

        flat = _core.model_gravity_matrix_numpy(
            np.asarray(coords[:, 0], dtype=float),
            np.asarray(coords[:, 1], dtype=float),
            np.asarray(relevances, dtype=float),
            np.asarray(tot_outflows, dtype=float),
            self._deterrence_func_type,
            float(self._deterrence_func_args[0]),
            float(self._origin_exp),
            float(self._destination_exp),
            self._gravity_type,
            out_format,
        )
        od_matrix = np.asarray(flat, dtype=float).reshape((n_locs, n_locs))
        return self._from_matrix_to_flowdf(od_matrix, origins, spatial_tessellation)

    def _from_matrix_to_flowdf(self, flow_matrix, origins, spatial_tessellation):
        index2tileid = dict(enumerate(spatial_tessellation[self._tile_id_column].values))
        output_list = [
            [index2tileid[int(i)], index2tileid[j], flow]
            for i in origins
            for j, flow in enumerate(flow_matrix[int(i)])
            if flow > 0.0
        ]
        return flow_dataframe(output_list, tessellation=spatial_tessellation, tile_id=self._tile_id_column)

    def fit(self, flow_df, relevance_column=RELEVANCE):
        try:
            import statsmodels as sm
            from statsmodels.genmod.generalized_linear_model import GLM
        except ImportError as exc:
            raise ImportError("statsmodels is required: pip install skmob2[generation]") from exc

        if not hasattr(flow_df, "tessellation"):
            raise AttributeError("flow_df must expose a tessellation attribute to fit Gravity.")

        tessellation = to_pandas_frame(flow_df.tessellation)
        self.lats_lngs = tessellation_lat_lngs(tessellation)
        self.weights = tessellation[relevance_column].fillna(0).to_numpy(dtype=float)
        self.tileid2index = dict((tileid, i) for i, tileid in enumerate(tessellation[TILE_ID].values))
        self.X, self.y = [], []

        for _, flow_example in to_pandas_frame(flow_df).iterrows():
            self._update_training_set(flow_example)

        poisson_model = GLM(
            self.y, self.X, family=sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log())
        )
        poisson_results = poisson_model.fit()
        if self._gravity_type == "globally constrained":
            self._origin_exp = poisson_results.params[1]
            self._destination_exp = poisson_results.params[2]
            self._deterrence_func_args = [poisson_results.params[3]]
        else:
            self._origin_exp = 1.0
            self._destination_exp = poisson_results.params[-2]
            self._deterrence_func_args = [poisson_results.params[-1]]
        del self.X
        del self.y

    def _update_training_set(self, flow_example):
        id_origin = flow_example["origin"]
        id_destination = flow_example["destination"]
        trips = flow_example[FLOW]
        if id_origin == id_destination:
            return
        try:
            coords_origin = self.lats_lngs[self.tileid2index[id_origin]]
            weight_origin = self.weights[self.tileid2index[id_origin]]
            coords_destination = self.lats_lngs[self.tileid2index[id_destination]]
            weight_destination = self.weights[self.tileid2index[id_destination]]
        except KeyError:
            return
        if weight_destination <= 0:
            return
        dist = haversine_km(tuple(coords_origin), tuple(coords_destination))
        sc_vars = (
            [np.log(weight_origin)]
            if self._gravity_type == "globally constrained"
            else ci(self.tileid2index[id_origin], len(self.tileid2index))
        )
        if self._deterrence_func_type == "exponential":
            self.X += [[1.0] + sc_vars + [np.log(weight_destination), -dist]]
        else:
            self.X += [[1.0] + sc_vars + [np.log(weight_destination), np.log(dist)]]
        self.y += [float(trips)]
