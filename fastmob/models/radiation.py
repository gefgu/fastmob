from __future__ import annotations

import secrets

import narwhals as nw

from fastmob import _core
from fastmob.core import FlowDataFrame, Locations

RELEVANCE = "relevance"
TILE_ID = "tile_id"
TOT_OUTFLOW = "tot_outflow"


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
    Generation accepts :class:`~fastmob.core.Locations` with global location IDs
    and numeric centre coordinates.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.core import Locations
    >>> from fastmob.models import Radiation
    >>> locations = Locations(pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [1_000_000, 2_700_000, 880_000, 1_600_000],
    ...     "tot_outflow": [100_000, 250_000, 90_000, 150_000],
    ... }))
    >>> radiation = Radiation()
    >>> rad_flows = radiation.generate(locations, tot_outflows_column="tot_outflow",
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

    def generate(
        self,
        locations: Locations,
        tot_outflows_column=TOT_OUTFLOW,
        relevance_column=RELEVANCE,
        out_format="flows",
        random_state: int | None = None,
    ):
        """Generate synthetic flows with the Radiation model.

        Parameters
        ----------
        locations : Locations
            Global locations on which to run the model. Must include a relevance
            column; the total-outflow column is
            required when ``out_format`` is ``"flows"`` or ``"flows_sample"``.
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
            array indexed by location order.

        Raises
        ------
        KeyError
            If ``out_format`` requires a total-outflow column but it is absent from
            ``locations``.
        ValueError
            If ``out_format`` is not one of the accepted values.
        """
        if not isinstance(locations, Locations):
            raise TypeError("Spatial generation models require Locations(scope='global'); use Locations.from_tessellation(...) for legacy inputs")
        prepared = locations.model_input()

        if out_format not in ["flows", "flows_sample", "probabilities"]:
            raise ValueError(
                f'Value of out_format "{out_format}" is not valid. \nValid values: flows, flows_sample, probabilities.'
            )
        if random_state is not None and random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        if "flows" in out_format:
            if tot_outflows_column not in prepared.frame.columns:
                raise KeyError(
                    f"The column {tot_outflows_column} for the 'tot_outflows' must be present in the tessellation."
                )
            outflows = prepared.values(tot_outflows_column)
        else:
            outflows = nw.new_series("tot_outflow", [1.0] * len(prepared.frame), backend=prepared.frame.implementation).cast(nw.Float64).to_arrow()
        if out_format == "flows_sample":
            origins, destinations, values = _core.model_radiation_sample_flows_arrow(
                prepared.latitudes,
                prepared.longitudes,
                prepared.values(relevance_column),
                outflows,
                int(random_state) if random_state is not None else secrets.randbits(64),
            )
        else:
            origins, destinations, values = _core.model_radiation_flows_arrow(
                prepared.latitudes,
                prepared.longitudes,
                prepared.values(relevance_column),
                outflows,
                "flows" if out_format == "flows" else "probabilities",
            )
        ids = prepared.location_ids().to_list()
        origin_values = origins.to_pylist()
        destination_values = destinations.to_pylist()
        frame = nw.from_dict(
            {"origin": [ids[int(i)] for i in origin_values], "destination": [ids[int(i)] for i in destination_values], "flow": values},
            backend=prepared.frame.implementation,
        ).to_native()
        return FlowDataFrame(frame, locations=locations, tile_id=locations.location_id_col)
