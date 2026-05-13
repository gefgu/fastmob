from __future__ import annotations

import operator

import numpy as np

from ._common import (
    RELEVANCE,
    TILE_ID,
    TOT_OUTFLOW,
    flow_dataframe,
    haversine_km,
    tessellation_lat_lngs,
    to_pandas_frame,
)


def _core_radiation_probabilities(lats_lngs, relevances, tot_outflows):
    try:
        from skmob2 import _core
    except Exception:
        return None
    try:
        origins, destinations, probabilities = _core.model_radiation_probabilities(
            np.asarray(lats_lngs[:, 0], dtype=float),
            np.asarray(lats_lngs[:, 1], dtype=float),
            np.asarray(relevances, dtype=float),
            np.asarray(tot_outflows, dtype=float),
        )
        return origins, destinations, np.asarray(probabilities, dtype=float)
    except Exception:
        return None


class Radiation:
    """Radiation model compatible with original scikit-mobility generation APIs."""

    def __init__(self, name="Radiation model"):
        self.name_ = name
        self._spatial_tessellation = None
        self._out_format = None

    def _get_flows(self, origin, total_relevance):
        edges = []
        probs = []
        origin_lat, origin_lng = self.lats_lngs[origin]
        origin_relevance = self.relevances[origin]
        origin_outflow = getattr(self, "tot_outflows", np.ones(len(self.lats_lngs), dtype=int))[origin]

        if origin_outflow > 0.0:
            normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance)
            destinations_and_distances = []
            for destination, (dest_lat, dest_lng) in enumerate(self.lats_lngs):
                if destination != origin:
                    destinations_and_distances.append(
                        (destination, haversine_km((origin_lat, origin_lng), (dest_lat, dest_lng)))
                    )
            destinations_and_distances.sort(key=operator.itemgetter(1))

            sum_inside = 0.0
            for destination, _ in destinations_and_distances:
                destination_relevance = self.relevances[destination]
                prob = (
                    normalization_factor
                    * (origin_relevance * destination_relevance)
                    / ((origin_relevance + sum_inside) * (origin_relevance + sum_inside + destination_relevance))
                )
                sum_inside += destination_relevance
                edges.append([origin, destination])
                probs.append(prob)

            probs = np.asarray(probs, dtype=float)
            if self._out_format == "flows":
                quantities = np.rint(origin_outflow * probs)
            elif self._out_format == "flows_sample":
                quantities = np.random.multinomial(int(origin_outflow), probs)
            else:
                quantities = probs
            edges = [edges[i] + [od] for i, od in enumerate(quantities)]
        return edges

    def generate(
        self,
        spatial_tessellation,
        tile_id_column=TILE_ID,
        tot_outflows_column=TOT_OUTFLOW,
        relevance_column=RELEVANCE,
        out_format="flows",
    ):
        spatial_tessellation = to_pandas_frame(spatial_tessellation)
        self._out_format = out_format
        self._tile_id_column = tile_id_column
        self.lats_lngs = tessellation_lat_lngs(spatial_tessellation)
        self.relevances = spatial_tessellation[relevance_column].fillna(0).to_numpy(dtype=float)
        if "flows" in out_format:
            if tot_outflows_column not in spatial_tessellation.columns:
                raise KeyError(
                    "The column %s for the 'tot_outflows' must be present in the tessellation." % tot_outflows_column
                )
            self.tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).to_numpy(dtype=int)
        if out_format not in ["flows", "flows_sample", "probabilities"]:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows, flows_sample, probabilities.'
                % out_format
            )

        if out_format != "flows_sample":
            outflows = self.tot_outflows if "flows" in out_format else np.ones(len(self.lats_lngs), dtype=float)
            core_result = _core_radiation_probabilities(self.lats_lngs, self.relevances, outflows)
            if core_result is not None:
                origins, destinations, probabilities = core_result
                quantities = (
                    np.rint(outflows[np.asarray(origins, dtype=int)] * probabilities)
                    if out_format == "flows"
                    else probabilities
                )
                all_flows = [
                    [int(origin), int(destination), quantity]
                    for origin, destination, quantity in zip(origins, destinations, quantities)
                ]
                return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)

        total_relevance = np.sum(self.relevances)
        all_flows = []
        for origin in range(len(spatial_tessellation)):
            all_flows.extend(self._get_flows(origin, total_relevance))
        return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)

    def _from_matrix_to_flowdf(self, all_flows, spatial_tessellation):
        index2tileid = dict(enumerate(spatial_tessellation[self._tile_id_column].values))
        output_list = [[index2tileid[i], index2tileid[j], flow] for i, j, flow in all_flows if flow > 0.0]
        return flow_dataframe(output_list, tessellation=spatial_tessellation, tile_id=self._tile_id_column)
