from __future__ import annotations

import operator

import numpy as np

from skmob2 import _core

from ._common import (
    RELEVANCE,
    TILE_ID,
    TOT_OUTFLOW,
    flow_dataframe,
    haversine_km,
    tessellation_lat_lngs,
    to_pandas_frame,
)


class Radiation:
    """Radiation model compatible with original scikit-mobility generation APIs."""

    def __init__(self, name="Radiation model"):
        self.name_ = name

    def _get_flows(self, origin, total_relevance):
        """Reference implementation kept for test verification only."""
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
        self._tile_id_column = tile_id_column
        lats_lngs = tessellation_lat_lngs(spatial_tessellation)
        relevances = spatial_tessellation[relevance_column].fillna(0).to_numpy(dtype=float)

        if out_format not in ["flows", "flows_sample", "probabilities"]:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows, flows_sample, probabilities.'
                % out_format
            )
        if "flows" in out_format:
            if tot_outflows_column not in spatial_tessellation.columns:
                raise KeyError(
                    "The column %s for the 'tot_outflows' must be present in the tessellation."
                    % tot_outflows_column
                )
            tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).to_numpy(dtype=int)
            outflows = tot_outflows.astype(float)
        else:
            tot_outflows = None
            outflows = np.ones(len(lats_lngs), dtype=float)

        origins_arr, destinations_arr, probabilities_arr = _core.model_radiation_probabilities(
            np.asarray(lats_lngs[:, 0], dtype=float),
            np.asarray(lats_lngs[:, 1], dtype=float),
            np.asarray(relevances, dtype=float),
            outflows,
        )
        origins_arr = np.asarray(origins_arr, dtype=int)
        destinations_arr = np.asarray(destinations_arr, dtype=int)
        probabilities_arr = np.asarray(probabilities_arr, dtype=float)

        if out_format == "flows_sample":
            quantities = np.zeros(len(origins_arr), dtype=float)
            for origin_idx in np.unique(origins_arr):
                mask = origins_arr == origin_idx
                count = int(tot_outflows[origin_idx])
                if count <= 0:
                    continue
                probs = probabilities_arr[mask]
                prob_sum = probs.sum()
                if prob_sum <= 0:
                    continue
                probs = probs / prob_sum
                quantities[mask] = np.random.multinomial(count, probs)
        elif out_format == "flows":
            quantities = np.rint(outflows[origins_arr] * probabilities_arr)
        else:
            quantities = probabilities_arr

        all_flows = [
            [int(o), int(d), q]
            for o, d, q in zip(origins_arr, destinations_arr, quantities)
            if q > 0.0
        ]
        return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)

    def _from_matrix_to_flowdf(self, all_flows, spatial_tessellation):
        index2tileid = dict(enumerate(spatial_tessellation[self._tile_id_column].values))
        output_list = [[index2tileid[i], index2tileid[j], flow] for i, j, flow in all_flows if flow > 0.0]
        return flow_dataframe(output_list, tessellation=spatial_tessellation, tile_id=self._tile_id_column)
