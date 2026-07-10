from __future__ import annotations

import operator

import numpy as np

from fkmob import _core

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
    """Radiation model.

    The radiation model for human migration. The model assumes that a traveller's
    destination choice follows two steps: each opportunity at every location is
    assigned a random fitness drawn from a distribution :math:`P(z)`, and the traveller
    accepts the closest opportunity whose fitness exceeds a threshold drawn from the
    same distribution. The resulting average flow from :math:`i` to :math:`j` is
    [SGMB2012]_:

    .. math::

        T_{ij} = O_i
        \\frac{1}{1 - \\frac{m_i}{M}}
        \\frac{m_i m_j}{(m_i + s_{ij})(m_i + m_j + s_{ij})}

    where :math:`O_i` is the total outflow from location :math:`i`, :math:`m_i`
    (:math:`m_j`) is the relevance (e.g. population) at the origin (destination),
    :math:`s_{ij}` is the total relevance within the circle of radius :math:`r_{ij}`
    centred on :math:`i` excluding origin and destination, and
    :math:`M = \\sum_i m_i` is the total system relevance.

    Parameters
    ----------
    name : str, optional
        Human-readable label for this model instance. The default is
        ``"Radiation model"``.

    Attributes
    ----------
    name : str
        Human-readable label of this model instance.

    Notes
    -----
    ``spatial_tessellation`` accepts any pandas-compatible eager dataframe. It must
    contain a ``tile_id`` column and either a ``geometry`` column or explicit ``lat`` /
    ``lng`` columns from which tile centroids are derived.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fkmob.models import Radiation
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [1_000_000, 2_700_000, 880_000, 1_600_000],
    ...     "tot_outflow": [100_000, 250_000, 90_000, 150_000],
    ... })
    >>> np.random.seed(0)
    >>> radiation = Radiation()
    >>> rad_flows = radiation.generate(tessellation, tile_id_column="tile_id",
    ...                                tot_outflows_column="tot_outflow",
    ...                                relevance_column="relevance",
    ...                                out_format="flows_sample")
    >>> print(rad_flows.head())

    References
    ----------
    .. [SGMB2012] Simini, F., González, M. C., Maritan, A. & Barabási, A.-L. (2012).
       A universal model for mobility and migration patterns. *Nature*, 484(7392),
       96–100.
    .. [MSJB2013] Masucci, A. P., Serras, J., Johansson, A. & Batty, M. (2013).
       Gravity versus radiation models: On the importance of scale and heterogeneity
       in commuting flows. *Physical Review E*, 88(2), 022812.

    See Also
    --------
    Gravity
    """

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
        """Generate synthetic flows with the Radiation model.

        Parameters
        ----------
        spatial_tessellation : DataFrame or GeoDataFrame
            The spatial tessellation on which to run the model. Must include a tile
            identifier column and a relevance column; the total-outflow column is
            required when ``out_format`` is ``"flows"`` or ``"flows_sample"``.
        tile_id_column : str, optional
            Name of the column containing the location identifier. The default is
            ``"tile_id"``.
        tot_outflows_column : str, optional
            Name of the column containing the total outflow per location. Required
            when ``out_format`` is ``"flows"`` or ``"flows_sample"``. The default is
            ``"tot_outflow"``.
        relevance_column : str, optional
            Name of the column containing the location relevance (e.g. population).
            The default is ``"relevance"``.
        out_format : str, optional
            Format of the output. Accepted values:

            - ``"flows"`` — expected (average) flows between each pair of locations.
            - ``"flows_sample"`` — randomly sampled integer flows drawn from a
              multinomial distribution.
            - ``"probabilities"`` — raw trip probability from each origin to each
              destination (does not require ``tot_outflows_column``).

            The default is ``"flows"``.

        Returns
        -------
        DataFrame
            A dataframe with columns ``origin``, ``destination``, and ``flow``
            (or ``probability``). A ``to_matrix()`` method converts this to a 2-D
            numpy array indexed by tessellation order.

        Raises
        ------
        KeyError
            If ``out_format`` requires a total-outflow column but it is absent from
            ``spatial_tessellation``.
        ValueError
            If ``out_format`` is not one of the accepted values.
        """
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
                    "The column %s for the 'tot_outflows' must be present in the tessellation." % tot_outflows_column
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

        all_flows = [[int(o), int(d), q] for o, d, q in zip(origins_arr, destinations_arr, quantities) if q > 0.0]
        return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)

    def _from_matrix_to_flowdf(self, all_flows, spatial_tessellation):
        index2tileid = dict(enumerate(spatial_tessellation[self._tile_id_column].values))
        output_list = [[index2tileid[i], index2tileid[j], flow] for i, j, flow in all_flows if flow > 0.0]
        return flow_dataframe(output_list, tessellation=spatial_tessellation, tile_id=self._tile_id_column)
