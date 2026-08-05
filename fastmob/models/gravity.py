from __future__ import annotations

import narwhals as nw
import numpy as np

from fastmob import _core
from fastmob.core import FlowDataFrame, Locations

FLOW = "flow"
RELEVANCE = "relevance"
TILE_ID = "tile_id"
TOT_OUTFLOW = "tot_outflow"


def ci(i, number_locs):
    """Return a one-hot normalisation indicator for the singly constrained Poisson fit.

    The singly constrained multinomial fit is implemented as a Poisson regression.
    Each origin location requires one extra indicator column to enforce its
    normalisation constraint. This function returns that indicator vector for
    origin ``i``.

    Parameters
    ----------
    i : int
        Index of the origin location.
    number_locs : int
        Total number of locations in the tessellation.

    Returns
    -------
    list of float
        A list of length ``number_locs`` that is ``1.0`` at position ``i`` and
        ``0.0`` elsewhere.
    """
    c = list(np.zeros(number_locs))
    c[i] = 1.0
    return c


class Gravity:
    """Gravity model.

    The Gravity model of human migration. In its original formulation, the probability
    :math:`T_{ij}` of moving from location :math:`i` to location :math:`j` is
    proportional to [Z1946]_:

    .. math::

        T_{ij} \\propto \\frac{P_i P_j}{r_{ij}}

    where :math:`P_i` and :math:`P_j` are the populations (or relevances) of locations
    :math:`i` and :math:`j` and :math:`r_{ij}` is the distance between them. The general
    form is [BBGJLLMRST2018]_:

    .. math::

        T_{ij} = K m_i m_j f(r_{ij})

    where :math:`K` is a constant and :math:`f(r_{ij})` is a *deterrence function* — a
    decreasing function of distance commonly modelled as a power law or exponential.

    **Singly constrained variant.** When the total outflow :math:`O_i` from each origin
    is known, the model estimates destinations only:

    .. math::

        T_{ij} = O_i \\frac{m_j f(r_{ij})}{\\sum_k m_k f(r_{ik})}.

    **Globally constrained variant.** Both origin and destination totals are fixed via
    balancing factors :math:`K_i` and :math:`L_j`:

    .. math::

        T_{ij} = K_i O_i L_j D_j f(r_{ij}).

    Parameters
    ----------
    deterrence_func_type : str, optional
        The deterrence function to use. Accepted values: ``"power_law"`` and
        ``"exponential"``. The default is ``"power_law"``.
    deterrence_func_args : list, optional
        Arguments for the deterrence function. For ``"power_law"`` this is the
        exponent; for ``"exponential"`` it is the decay rate. The default is
        ``[-2.0]``.
    origin_exp : float, optional
        Exponent applied to the origin's relevance (only used in the globally
        constrained model). The default is ``1.0``.
    destination_exp : float, optional
        Exponent applied to the destination's relevance. The default is ``1.0``.
    gravity_type : str, optional
        The model variant. Accepted values: ``"singly constrained"`` and
        ``"globally constrained"``. The default is ``"singly constrained"``.
    name : str, optional
        Human-readable label for this model instance. The default is
        ``"Gravity model"``.

    Attributes
    ----------
    deterrence_func_type : str
        The deterrence function in use.
    deterrence_func_args : list
        Arguments for the deterrence function.
    origin_exp : float
        Exponent applied to the origin's relevance.
    destination_exp : float
        Exponent applied to the destination's relevance.
    gravity_type : str
        The model variant (``"singly constrained"`` or ``"globally constrained"``).
    name : str
        Human-readable label of this model instance.

    Notes
    -----
    ``spatial_tessellation`` accepts any pandas-compatible eager dataframe. It must
    contain a ``tile_id`` column (or whichever name is passed as ``tile_id_column``)
    and either a ``geometry`` column (GeoPandas GeoDataFrame) or explicit ``lat`` /
    ``lng`` columns from which tile centroids are derived.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fastmob.models import Gravity
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [1_000_000, 2_700_000, 880_000, 1_600_000],
    ...     "tot_outflow": [100_000, 250_000, 90_000, 150_000],
    ... })
    >>> np.random.seed(0)
    >>> gravity = Gravity(gravity_type="singly constrained")
    >>> print(gravity)
    Gravity(name="Gravity model", deterrence_func_type="power_law", deterrence_func_args=[-2.0], origin_exp=1.0, destination_exp=1.0, gravity_type="singly constrained")
    >>> fdf = gravity.generate(tessellation, tile_id_column="tile_id",
    ...                        tot_outflows_column="tot_outflow",
    ...                        relevance_column="relevance", out_format="flows")
    >>> print(fdf.head())

    References
    ----------
    .. [Z1946] Zipf, G. K. (1946). The P1 P2/D Hypothesis: On the Intercity Movement
       of Persons. *American Sociological Review*, 11(6), 677–686.
    .. [W1971] Wilson, A. G. (1971). A family of spatial interaction models, and
       associated developments. *Environment and Planning A*, 3(1), 1–32.
    .. [BBGJLLMRST2018] Barbosa, H. et al. (2018). Human mobility: Models and
       applications. *Physics Reports*, 734, 1–74.

    See Also
    --------
    Radiation
    """

    def __init__(
        self,
        deterrence_func_type="power_law",
        deterrence_func_args=None,
        origin_exp=1.0,
        destination_exp=1.0,
        gravity_type="singly constrained",
        name="Gravity model",
    ):
        if deterrence_func_args is None:
            deterrence_func_args = [-2.0]
        self._name = name
        self._deterrence_func_args = deterrence_func_args
        self._origin_exp = origin_exp
        self._destination_exp = destination_exp
        self._gravity_type = gravity_type
        if deterrence_func_type not in ("power_law", "exponential"):
            print(
                f'Deterrence function type "{deterrence_func_type}" not available. Power law will be used.\n'
                "Available deterrence functions are [power_law, exponential]"
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
            f'Gravity(name="{self._name}", deterrence_func_type="{self._deterrence_func_type}", deterrence_func_args={self._deterrence_func_args}, '
            f'origin_exp={self._origin_exp}, destination_exp={self._destination_exp}, gravity_type="{self._gravity_type}")'
        )

    def generate(
        self,
        locations: Locations,
        tot_outflows_column=TOT_OUTFLOW,
        relevance_column=RELEVANCE,
        out_format="flows",
        random_state: int | None = None,
    ):
        """Generate synthetic flows with the Gravity model.

        Parameters
        ----------
        spatial_tessellation : DataFrame or GeoDataFrame
            The spatial tessellation on which to run the model. Must include a tile
            identifier column, a relevance column, and (for ``"flows"`` and
            ``"flows_sample"`` output) a total-outflow column.
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
            - ``"flows_sample"`` — randomly sampled integer flows.
            - ``"probabilities"`` — probability of a trip from each origin to each
              destination (does not require ``tot_outflows_column``).

            The default is ``"flows"``.

        Returns
        -------
        DataFrame
            A dataframe with columns ``origin``, ``destination``, and ``flow``
            (or ``probability``). A ``to_matrix()`` method converts this to a 2-D
            numpy array.

        Raises
        ------
        KeyError
            If ``out_format`` requires a total-outflow column but it is absent from
            ``spatial_tessellation``.
        """
        if not isinstance(locations, Locations):
            raise TypeError(
                "Spatial generation models require Locations(scope='global'); use Locations.from_tessellation(...) for legacy inputs"
            )
        prepared = locations.model_input()
        n_locs = len(prepared.frame)

        if out_format not in ["flows", "flows_sample", "probabilities"]:
            print(
                f'Output format "{out_format}" not available. Flows will be used.\n'
                "Available output formats are [flows, flows_sample, probabilities]"
            )
            out_format = "flows"

        if "flows" in out_format:
            if tot_outflows_column not in prepared.frame.columns:
                raise KeyError("The column 'tot_outflows' must be present in the tessellation.")
            tot_outflows = prepared.values(tot_outflows_column)
        else:
            tot_outflows = (
                nw.new_series("tot_outflow", [0.0] * n_locs, backend=prepared.frame.implementation)
                .cast(nw.Float64)
                .to_arrow()
            )

        origins, destinations, values = _core.model_gravity_flows_arrow(
            prepared.latitudes,
            prepared.longitudes,
            prepared.values(relevance_column),
            tot_outflows,
            self._deterrence_func_type,
            float(self._deterrence_func_args[0]),
            float(self._origin_exp),
            float(self._destination_exp),
            self._gravity_type,
            out_format,
        )
        if out_format == "flows_sample":
            rng = np.random.default_rng(random_state)
            origin_values = np.asarray(origins, dtype=np.int64)
            probability_values = np.asarray(values, dtype=float)
            quantities = np.zeros_like(probability_values)
            outflow_values = np.asarray(tot_outflows, dtype=np.int64)
            for origin in np.unique(origin_values):
                mask = origin_values == origin
                quantities[mask] = rng.multinomial(
                    int(outflow_values[origin]),
                    probability_values[mask] / probability_values[mask].sum(),
                )
            values = nw.new_series("flow", quantities, backend=prepared.frame.implementation).to_arrow()
        ids = prepared.location_ids().to_list()
        frame = nw.from_dict(
            {
                "origin": [ids[int(i)] for i in np.asarray(origins)],
                "destination": [ids[int(i)] for i in np.asarray(destinations)],
                "flow": values,
            },
            backend=prepared.frame.implementation,
        ).to_native()
        return FlowDataFrame(frame, locations=locations, tile_id=locations.location_id_col)

    def fit(self, flow_df, relevance_column=RELEVANCE):
        """Fit the Gravity model parameters to observed flows via Poisson regression.

        Uses a Generalised Linear Model (GLM) with a Poisson family and log link to
        estimate ``destination_exp`` and ``deterrence_func_args`` (and
        ``origin_exp`` for the globally constrained variant) from real flows
        [FM1982]_.

        Parameters
        ----------
        flow_df : DataFrame with tessellation attribute
            Observed flows. Must expose a ``.tessellation`` attribute (e.g. the
            ``FlowDataFrame`` returned by ``Gravity.generate()`` or
            ``Radiation.generate()``) and contain ``origin``, ``destination``, and
            ``flow`` columns.
        relevance_column : str, optional
            Name of the column in ``flow_df.tessellation`` that holds the location
            relevance (e.g. population). The default is ``"relevance"``.

        Notes
        -----
        Modifies ``deterrence_func_args``, ``destination_exp``, and (for the globally
        constrained variant) ``origin_exp`` in-place.

        References
        ----------
        .. [FM1982] Flowerdew, R. & Murray, A. (1982). A method of fitting the gravity
           model based on the Poisson distribution. *Journal of Regional Science*,
           22(2), 191–202.
        """
        try:
            import statsmodels as sm
            from statsmodels.genmod.generalized_linear_model import GLM
        except ImportError as exc:
            raise ImportError("statsmodels is required: pip install fastmob[generation]") from exc

        if not hasattr(flow_df, "locations") or flow_df.locations is None:
            raise AttributeError("flow_df must expose a Locations catalogue to fit Gravity.")
        prepared = flow_df.locations.model_input()
        self.lats_lngs = np.column_stack((np.asarray(prepared.latitudes), np.asarray(prepared.longitudes)))
        self.weights = np.asarray(prepared.values(relevance_column), dtype=float)
        self.tileid2index = {tileid: i for i, tileid in enumerate(prepared.location_ids().to_list())}
        self.X, self.y = [], []

        for flow_example in nw.from_native(flow_df.df, eager_only=True).iter_rows(named=True):
            self._update_training_set(flow_example)

        poisson_model = GLM(
            self.y,
            self.X,
            family=sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log()),
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
        dist = float(_core.haversine_km(*coords_origin, *coords_destination))
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
