from __future__ import annotations

import math
import warnings

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

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


class _PoissonResults:
    """The few attributes of a statsmodels GLM result that `Gravity` reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def conf_int(self):
        return self._interval


def _concentrated_poisson(design, response, origin, robust):
    """Poisson regression with one free level per origin, without building the origin dummies.

    For a log-linear Poisson model with an origin effect, the effect has a closed form given the other
    coefficients, ``exp(alpha_i) = O_i / sum_j exp(x_ij b)``, where ``O_i`` is the total flow of origin ``i``. The
    remaining coefficients maximise the concentrated (multinomial) likelihood, found here by Newton's method
    on arrays of one row per pair. The estimates and the heteroskedasticity-robust (HC0) errors equal those of the
    full Poisson regression on origin dummies, which needs memory growing with the cube of the number of places.
    """
    order = np.argsort(origin, kind="stable")
    x, y, o = design[order], response[order], origin[order]
    _, start, counts = np.unique(o, return_index=True, return_counts=True)
    totals = np.add.reduceat(y, start)
    keep = np.repeat(totals > 0, counts)
    if not keep.all():  # an origin whose observed flows are all zero carries no information
        x, y, o = x[keep], y[keep], o[keep]
        _, start, counts = np.unique(o, return_index=True, return_counts=True)
        totals = np.add.reduceat(y, start)
    k = x.shape[1]
    repeat_totals = np.repeat(totals, counts)

    def shares(beta):
        eta = x @ beta
        top = np.maximum.reduceat(eta, start)
        e = np.exp(eta - np.repeat(top, counts))
        norm = np.add.reduceat(e, start)
        return eta, e / np.repeat(norm, counts), top + np.log(norm)

    def log_likelihood(beta):
        eta, _, log_norm = shares(beta)
        return float(y @ eta - totals @ log_norm)

    beta = np.zeros(k)
    current = log_likelihood(beta)
    for _ in range(200):
        _, p, _ = shares(beta)
        mean = np.add.reduceat(p[:, None] * x, start) if k else None
        centered = x - np.repeat(mean, counts, axis=0)
        gradient = x.T @ y - totals @ mean
        information = (centered * (repeat_totals * p)[:, None]).T @ centered
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The gravity design matrix is rank-deficient: too few distinct origins or distances to fit"
            ) from exc
        scale = 1.0
        while scale > 1e-8 and log_likelihood(beta + scale * step) < current - 1e-12:
            scale /= 2.0
        beta = beta + scale * step
        new = log_likelihood(beta)
        done = abs(new - current) < 1e-12 * (1.0 + abs(new)) and np.abs(scale * step).max() < 1e-9
        current = new
        if done:
            break

    _, p, _ = shares(beta)
    mu = repeat_totals * p
    mean = np.add.reduceat(p[:, None] * x, start)
    centered = x - np.repeat(mean, counts, axis=0)
    information = (centered * mu[:, None]).T @ centered
    inverse = np.linalg.inv(information)
    if robust:
        scores = centered * (y - mu)[:, None]
        covariance = inverse @ (scores.T @ scores) @ inverse
    else:
        covariance = inverse
    bse = np.sqrt(np.diag(covariance))
    z = beta / bse
    pvalues = np.array([math.erfc(abs(v) / math.sqrt(2.0)) for v in z])
    interval = np.column_stack([beta - 1.959964 * bse, beta + 1.959964 * bse])

    def deviance(fitted):
        positive = y > 0
        term = np.zeros_like(y)
        term[positive] = y[positive] * np.log(y[positive] / fitted[positive])
        return float(2.0 * np.sum(term - (y - fitted)))

    unique, inverse_index = np.unique(y, return_inverse=True)
    log_factorial = np.array([math.lgamma(v + 1.0) for v in unique])[inverse_index]
    llf = float(np.sum(y * np.log(mu) - mu - log_factorial))
    return _PoissonResults(
        params=beta,
        bse=bse,
        pvalues=pvalues,
        _interval=interval,
        deviance=deviance(mu),
        null_deviance=deviance(np.full_like(y, y.mean())),
        nobs=len(y),
        llf=llf,
        pearson_chi2=float(np.sum((y - mu) ** 2 / mu)),
        df_resid=len(y) - k - len(totals),
    )


class Gravity:
    """Gravity model: flows between places from their sizes and the distance between them.

    The Gravity model of human migration. In its original formulation, the probability
    :math:`T_{ij}` of moving from location :math:`i` to location :math:`j` is
    proportional to [Z1946]:

    .. math::

        T_{ij} \\propto \\frac{P_i P_j}{r_{ij}}

    where :math:`P_i` and :math:`P_j` are the populations (or relevances) of locations
    :math:`i` and :math:`j` and :math:`r_{ij}` is the distance between them. The general
    form is [BBGJLLMRST2018]:

    .. math::

        T_{ij} = K m_i m_j f(r_{ij})

    where :math:`K` is a constant and :math:`f(r_{ij})` is a *deterrence function* — a
    decreasing function of distance commonly modelled as a power law or exponential.

    **Singly constrained variant.** When the total outflow :math:`O_i` from each origin
    is known, the model estimates destinations only:

    .. math::

        T_{ij} = O_i \\frac{m_j f(r_{ij})}{\\sum_k m_k f(r_{ik})}.

    **Globally constrained variant.** Only the total flow of the whole system is fixed, through a
    single constant :math:`K`:

    .. math::

        T_{ij} = K m_i^{a} m_j^{b} f(r_{ij}), \\qquad \\sum_{ij} T_{ij} = \\sum_i O_i.

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
    The places are given as a ``Locations`` catalogue (``Locations.from_tessellation`` builds one from a table with
    a tile id, latitude, longitude, a relevance column and, to generate flows, a total outflow column).

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.core import Locations
    >>> from fastmob.models import Gravity
    >>> tessellation = pd.DataFrame({
    ...     "tile_id": [0, 1, 2, 3],
    ...     "lat": [40.71, 41.88, 37.77, 33.44],
    ...     "lng": [-74.01, -87.62, -122.42, -112.07],
    ...     "relevance": [1_000_000, 2_700_000, 880_000, 1_600_000],
    ...     "tot_outflow": [100_000, 250_000, 90_000, 150_000],
    ... })
    >>> locations = Locations.from_tessellation(tessellation, tile_id_col="tile_id", lat_col="lat", lng_col="lng")
    >>> gravity = Gravity(gravity_type="singly constrained")
    >>> print(gravity)
    Gravity(name="Gravity model", deterrence_func_type="power_law", deterrence_func_args=[-2.0], origin_exp=1.0, destination_exp=1.0, gravity_type="singly constrained")
    >>> flows = gravity.generate(locations, out_format="flows")
    >>> int(round(flows.to_pandas()["flow"].sum()))
    590000

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
        locations : Locations
            Global locations on which to run the model. They must provide a
            relevance column and, for ``"flows"`` and ``"flows_sample"`` output,
            a total-outflow column.
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
            ``locations``.
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
        ids_arrow = pa.array(prepared.location_ids().to_arrow())
        frame = nw.from_dict(
            {
                "origin": pc.take(ids_arrow, pa.array(origins)),
                "destination": pc.take(ids_arrow, pa.array(destinations)),
                "flow": pa.array(values),
            },
            backend=prepared.frame.implementation,
        ).to_native()
        return FlowDataFrame(frame, locations=locations, tile_id=locations.location_id_col)

    def fit(self, flow_df, relevance_column=RELEVANCE, robust=True, include_zeros=True):
        """Fit the Gravity model parameters to observed flows via Poisson regression.

        Uses a Generalised Linear Model (GLM) with a Poisson family and log link to
        estimate ``destination_exp`` and ``deterrence_func_args`` (and
        ``origin_exp`` for the globally constrained variant) from real flows
        [FM1982].

        Parameters
        ----------
        flow_df : FlowDataFrame with a locations catalogue
            Observed flows. Must expose a ``.locations`` catalogue (for example a ``FlowDataFrame`` made with
            ``locations=...``, or the one returned by ``Gravity.generate()``) and contain ``origin``,
            ``destination`` and ``flow`` columns. List only the pairs with flow: the pairs without flow are
            added as zeros (see ``include_zeros``).
        relevance_column : str, optional
            Name of the column of the locations catalogue that holds the location
            relevance (e.g. population). The default is ``"relevance"``.
        robust : bool, optional
            Report heteroskedasticity-robust (sandwich) standard errors and confidence intervals. Real flows are
            usually overdispersed relative to the Poisson model, which makes the model-based errors too small;
            the point estimates are the same either way. Default True.
        include_zeros : bool, optional
            Add every pair of locations that is missing from ``flow_df`` as a flow of 0. Tables of observed flows,
            such as the output of `trajectory_to_od`, list only the pairs with flow, and fitting those alone
            biases the distance coefficient towards zero because the long pairs that nobody travels are missing
            (Santos Silva and Tenreyro, 2006). The added pairs are the off-diagonal pairs of the catalogue whose
            origin has at least one observed flow. The singly constrained fit needs memory for one row per pair (the square
            of the number of locations: about 400 MB and 10 seconds for 1,000 locations); pass False only when the table already lists
            every pair, as complete matrices do.
            Default True.

        Raises
        ------
        ImportError
            If statsmodels is not installed (only the globally constrained variant needs it).
        AttributeError
            If ``flow_df`` has no ``locations`` catalogue.
        ValueError
            If the design matrix is rank-deficient (too few distinct origins or distances).

        Notes
        -----
        Modifies ``deterrence_func_args``, ``destination_exp``, and (for the globally
        constrained variant) ``origin_exp`` in-place.

        After fitting, `fit_summary` returns the estimates with standard errors, 95% confidence intervals and
        p-values, and `fit_statistics` the number of flows used, the log-likelihood, the deviance, the
        pseudo R-squared and the dispersion. For ``"power_law"`` the distance coefficient is the exponent
        ``gamma`` of ``flow ∝ distance^gamma`` (negative: flows fall with distance); for ``"exponential"``
        the table reports the decay rate ``lambda`` of ``exp(-lambda * distance)`` in km^-1 (positive: flows fall with
        distance). Relevance that is itself made of the flows (for example check-ins counted in the destination)
        makes the destination exponent partly circular; prefer an independent size such as population or jobs.

        References
        ----------
        - [FM1982] Flowerdew, R. & Murray, A. (1982). A method of fitting the gravity model based on the Poisson
          distribution. *Journal of Regional Science*, 22(2), 191–202.
        - [ST2006] Santos Silva, J. M. C. & Tenreyro, S. (2006). The log of gravity. *The Review of Economics and
          Statistics*, 88(4), 641–658.

        Examples
        --------
        >>> import pandas as pd
        >>> from fastmob.core import Locations
        >>> from fastmob.models import Gravity
        >>> places = pd.DataFrame({
        ...     "tile_id": range(6),
        ...     "lat": [45.0, 45.1, 45.3, 45.2, 45.6, 45.5],
        ...     "lng": [7.0, 7.4, 7.1, 7.9, 7.6, 7.2],
        ...     "relevance": [9_000, 7_000, 5_000, 3_000, 8_000, 6_000],
        ...     "tot_outflow": [60_000, 50_000, 40_000, 30_000, 70_000, 45_000],
        ... })
        >>> locations = Locations.from_tessellation(places, tile_id_col="tile_id", lat_col="lat", lng_col="lng")
        >>> observed = Gravity(deterrence_func_args=[-2.0]).generate(locations, out_format="flows")
        >>> model = Gravity()
        >>> model.fit(observed)
        >>> round(model.deterrence_func_args[0], 2)
        -2.0
        >>> model.fit_summary().column("parameter").to_pylist()
        ['destination_exp', 'distance_coefficient']
        """
        singly = self._gravity_type != "globally constrained"
        if not singly:
            try:
                import statsmodels as sm
                from statsmodels.genmod.generalized_linear_model import GLM
            except ImportError as exc:
                raise ImportError(
                    "Gravity.fit with the globally constrained variant requires statsmodels. "
                    "Install it with `pip install statsmodels`."
                ) from exc

        if not hasattr(flow_df, "locations") or flow_df.locations is None:
            raise AttributeError("flow_df must expose a Locations catalogue to fit Gravity.")
        prepared = flow_df.locations.model_input()
        self.lats_lngs = np.column_stack((np.asarray(prepared.latitudes), np.asarray(prepared.longitudes)))
        self.weights = np.asarray(prepared.values(relevance_column), dtype=float)
        self.tileid2index = {tileid: i for i, tileid in enumerate(prepared.location_ids().to_list())}
        self.X, self.y, self._pairs = [], [], []

        for flow_example in nw.from_native(flow_df.df, eager_only=True).iter_rows(named=True):
            self._update_training_set(flow_example)
        if not self._pairs:
            raise ValueError("flow_df has no usable origin-destination pair to fit")

        design = np.asarray(self.X, dtype=float)
        response = np.asarray(self.y, dtype=float)
        origin = np.asarray([pair[0] for pair in self._pairs])
        if include_zeros:
            extra, extra_origin = self._zero_rows()
            if len(extra):
                design = np.vstack([design, extra])
                response = np.concatenate([response, np.zeros(len(extra))])
                origin = np.concatenate([origin, extra_origin])
        if singly:
            results = _concentrated_poisson(design, response, origin, robust)
            self._store_fit(results, robust)
            self._origin_exp = 1.0
            self._destination_exp = results.params[0]
            self._deterrence_func_args = [results.params[1]]
        else:
            if np.linalg.matrix_rank(design) < design.shape[1]:
                raise ValueError(
                    "The gravity design matrix is rank-deficient: too few distinct origins or distances to fit"
                )
            poisson_model = GLM(
                response,
                design,
                family=sm.genmod.families.family.Poisson(
                    link=getattr(sm.genmod.families.links, "Log", sm.genmod.families.links.log)()
                ),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
                results = poisson_model.fit(cov_type="HC0") if robust else poisson_model.fit()
            self._store_fit(results, robust)
            self._origin_exp = results.params[1]
            self._destination_exp = results.params[2]
            self._deterrence_func_args = [results.params[3]]
        del self.X
        del self.y
        del self._pairs

    def _zero_rows(self):
        """Design rows of the pairs with no observed flow, among origins that have at least one."""
        n = len(self.weights)
        seen = np.asarray(self._pairs)
        missing = np.ones((n, n), dtype=bool)
        missing[seen[:, 0], seen[:, 1]] = False
        np.fill_diagonal(missing, False)
        has_flow = np.zeros(n, dtype=bool)
        has_flow[seen[:, 0]] = True
        missing &= has_flow[:, None] & (self.weights > 0)[None, :]
        if self._gravity_type == "globally constrained":
            missing &= (self.weights > 0)[:, None]
        origin, destination = np.nonzero(missing)
        lat1, lng1 = np.radians(self.lats_lngs[origin]).T
        lat2, lng2 = np.radians(self.lats_lngs[destination]).T
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
        distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        keep = distance > 0
        origin, destination, distance = origin[keep], destination[keep], distance[keep]
        distance_term = -distance if self._deterrence_func_type == "exponential" else np.log(distance)
        destination_term = np.log(self.weights[destination])
        if self._gravity_type == "globally constrained":
            return (
                np.column_stack([np.ones(len(origin)), np.log(self.weights[origin]), destination_term, distance_term]),
                origin,
            )
        return np.column_stack([destination_term, distance_term]), origin

    def _store_fit(self, results, robust):
        """Keep the summary table and the fit statistics of the Poisson regression."""
        if self._gravity_type == "globally constrained":
            names, positions = ["intercept", "origin_exp", "destination_exp", "distance_coefficient"], [0, 1, 2, 3]
        else:
            last = len(results.params) - 1
            names, positions = ["destination_exp", "distance_coefficient"], [last - 1, last]
        interval = np.asarray(results.conf_int())
        rows = {"parameter": [], "estimate": [], "std_error": [], "ci_low": [], "ci_high": [], "p_value": []}
        for name, i in zip(names, positions):
            decay = name == "distance_coefficient" and self._deterrence_func_type == "exponential"
            low, high = interval[i, 0], interval[i, 1]
            rows["parameter"].append("distance_decay" if decay else name)
            rows["estimate"].append(float(results.params[i]))
            rows["std_error"].append(float(results.bse[i]))
            rows["ci_low"].append(float(low))
            rows["ci_high"].append(float(high))
            rows["p_value"].append(float(results.pvalues[i]))
        self._fit_summary = pa.table(rows)
        deviance, null = float(results.deviance), float(results.null_deviance)
        self._fit_statistics = {
            "n_observations": int(results.nobs),
            "log_likelihood": float(results.llf),
            "deviance": deviance,
            "pseudo_r_squared": 1.0 - deviance / null if null > 0 else float("nan"),
            "dispersion": float(results.pearson_chi2 / results.df_resid) if results.df_resid > 0 else float("nan"),
            "robust_errors": bool(robust),
        }

    def fit_summary(self):
        """The fitted parameters with standard errors, confidence intervals and p-values.

        Returns
        -------
        pyarrow.Table
            One row per model parameter with ``parameter``, ``estimate``, ``std_error``, ``ci_low``,
            ``ci_high`` (95%) and ``p_value``. For the exponential deterrence function the distance row is
            ``distance_decay``, the decay rate lambda of ``exp(-lambda * distance)`` in km^-1; otherwise ``distance_coefficient`` is the exponent of
            the distance.

        Raises
        ------
        RuntimeError
            If `fit` has not been called.
        """
        if not hasattr(self, "_fit_summary"):
            raise RuntimeError("Call fit() before fit_summary()")
        return self._fit_summary

    def fit_statistics(self):
        """Goodness-of-fit statistics of the last fit.

        Returns
        -------
        dict
            ``n_observations`` (flows used), ``log_likelihood``, ``deviance``, ``pseudo_r_squared`` (one minus
            the ratio of the deviance to the null deviance), ``dispersion`` (Pearson chi-squared per degree of
            freedom; well above 1 means overdispersion) and ``robust_errors``.

        Raises
        ------
        RuntimeError
            If `fit` has not been called.
        """
        if not hasattr(self, "_fit_statistics"):
            raise RuntimeError("Call fit() before fit_statistics()")
        return self._fit_statistics

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
        if dist <= 0:
            return
        distance_term = -dist if self._deterrence_func_type == "exponential" else np.log(dist)
        if self._gravity_type == "globally constrained":
            self.X += [[1.0, np.log(weight_origin), np.log(weight_destination), distance_term]]
        else:  # the origin enters through its own level, see _concentrated_poisson
            self.X += [[np.log(weight_destination), distance_term]]
        self.y += [float(trips)]
        self._pairs.append((self.tileid2index[id_origin], self.tileid2index[id_destination]))
