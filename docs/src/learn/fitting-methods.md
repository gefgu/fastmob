# How the fits work

**Question.** How does Fastkit-Mobility estimate a distance law or a gravity model, how sure can I be of the result, and what must I report so that a reader can check it?

This page explains the estimators, their limits and a reporting checklist. To run them, open [Compare to mobility laws](compare-to-mobility-laws.md) for distributions and [Validate against census flows](validate-against-census-flows.md) for flows.

## Distributions of distances

`fit_distance_distribution` fits one family to positive values (jump lengths, trip lengths, radii of gyration) by maximum likelihood. `compare_distance_distributions` fits several and ranks them.

| Step | What is done | Why it matters |
| --- | --- | --- |
| Range | Values outside `[xmin, xmax]`, and zeros, are left out and counted in `n_excluded`. Default: smallest positive value to largest. | Estimates depend on the range. Zeros are common (repeated check-ins at one place) and have no place in a power law. Raise `xmin` to ignore GPS noise. |
| Grouping | The values are grouped into 200 geometric bins on the range. | The likelihood is computed from the bin counts, so a million values cost no more than a thousand. |
| Likelihood | Each family is normalised on the same range, and the multinomial log-likelihood of the bin counts is maximised. | Families are comparable: they explain the same data on the same range. |
| Uncertainty | The bin counts are resampled (bootstrap, 100 replicates by default) and the model refitted; the report has the standard error and the percentile interval of each parameter. | Gives an interval, not only a point estimate. |
| Fit quality | The KS distance between the empirical and the fitted cumulative distribution; with `n_gof` above 0, a parametric-bootstrap p-value. | See the caveat below. |
| Comparison | AIC and BIC, their difference to the best model, and the Akaike weight. | Tells whether a more flexible form is supported by the data. |

The four families are the power law `x^-beta`, the exponential `exp(-x / kappa)`, the log-normal, and the truncated power law `(x + r0)^-beta exp(-x / kappa)` of González et al. (2008).

### Reading the result

- **A parameter at its bound.** Each parameter is searched within a range. If an estimate lands on the edge, `at_bound` lists it and the data do not identify it. For the truncated power law, a `kappa` at its upper bound (ten times the largest value by default) means there is no cutoff in the data: the model is a shifted power law, and a fitted `kappa` of 10^8 km is not evidence for a cutoff. The interval of such a parameter is not meaningful; say that the cutoff is not identified.
- **Goodness of fit.** With thousands of observations every simple model is rejected, because the test detects tiny departures. Report the KS distance and the AIC difference, and look at the fit on the data, instead of reading the p-value alone. A small p-value does not mean the model is useless, and a large one does not prove it right.
- **Several families.** An AIC difference below about 2 means the data cannot choose. The Akaike weight is a relative support among the families you list, not a probability that the family is true.
- **The earlier fit.** `fit_values_to_truncated_powerlaw` fits a log histogram by a grid search and returns no uncertainty. It is kept for the plotting helpers; use `fit_distance_distribution` for a result that you report.

## Gravity models

`Gravity.fit` fits `flow ∝ origin^a * destination^b * f(distance)` by Poisson regression with a log link (Flowerdew and Murray, 1982), the standard estimator for flow counts (Santos Silva and Tenreyro, 2006, discuss its use for gravity models).

| Output | Meaning |
| --- | --- |
| `fit_summary()` | Estimate, standard error, 95% confidence interval and p-value of each parameter. The distance row is `distance_coefficient`, the exponent in `flow ∝ distance^gamma` (negative), or `distance_decay` for the exponential form, in km⁻¹. |
| `fit_statistics()` | Number of flows, log-likelihood, deviance, pseudo R-squared (one minus the deviance over the null deviance) and the dispersion. |

- **Robust errors.** Flow data are almost always overdispersed: their variance is larger than their mean, so the standard errors of a Poisson model are too small. The dispersion in `fit_statistics()` (Pearson chi-squared per degree of freedom) is 1 for a good Poisson fit and well above 1 for real flows (about 16 for the check-in flows in the notebook). `Gravity.fit` therefore reports heteroskedasticity-robust (sandwich) errors by default; the estimates do not change. Pass `robust=False` for the model-based errors.
- **Singly or globally constrained.** The singly constrained model has one free level for each origin, so it matches the outflow of every origin and the distance exponent is estimated from how flows spread among destinations. It is the right choice when the origin totals are observed (a census, or the origins of a sample). The globally constrained model has a single scale and fits only if the total flow of each origin follows the model.
- **Unweighted samples.** A bias that depends only on the origin (more people sampled in some home zones) does not change the distance exponent of a singly constrained fit, but it does change the flow levels. Weight the sample before comparing absolute flows (see the notebook).
- **Zones.** The exponent depends on the zone size and on the range of distances in the data. Report both.

## Comparing flows

`compare_flow_models` scores model flows against observed flows over the union of their origin-destination pairs.

| Score | Meaning |
| --- | --- |
| `cpc` | Common part of commuters: `2 * sum(min(observed, model)) / (sum(observed) + sum(model))`. 1 is identical. |
| `cpl` | Common part of links: the same for the sets of pairs that have a flow, whatever its size. |
| `r_squared`, `spearman` | Squared Pearson and rank correlation of the pair flows. |
| `poisson_deviance`, `pseudo_r_squared` | Fit of the model flows as the mean of Poisson counts, against a model that predicts the average flow everywhere. |

All of them depend on the zone size. CPC is dominated by the largest flows, so a high CPC can hide a poor fit to the many small ones: look at CPL and the rank correlation too.

## What to report

1. The data, the filtering, and the number of values or flows used.
2. For distributions: the family, the range `[xmin, xmax]` and `n_excluded`, the estimates with their intervals and the number of bootstrap replicates, the parameters at a bound, the KS distance (and the p-value with its replicates), and the comparison with the alternatives (AIC and weights).
3. For gravity fits: the deterrence function, the constraint type, the estimates with intervals, whether the errors are robust, the dispersion and the zone size.
4. For validation against a reference table: the weighting, the scores and the zone size, with the same scores for a trivial baseline (for example the unweighted sample).
5. The version of Fastkit-Mobility and the random seeds.

## References

- González, M. C., Hidalgo, C. A. & Barabási, A.-L. (2008). Understanding individual human mobility patterns. *Nature* 453, 779-782.
- Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review* 51, 661-703.
- Flowerdew, R. & Murray, A. (1982). A method of fitting the gravity model based on the Poisson distribution. *Journal of Regional Science* 22, 191-202.
- Santos Silva, J. M. C. & Tenreyro, S. (2006). The log of gravity. *The Review of Economics and Statistics* 88, 641-658.
- Simini, F., González, M. C., Maritan, A. & Barabási, A.-L. (2012). A universal model for mobility and migration patterns. *Nature* 484, 96-100.

## Next steps

- Fit and compare the distributions of your own jump lengths: [Compare to mobility laws](compare-to-mobility-laws.md).
- Check flows against a reference table: [Validate against census flows](validate-against-census-flows.md).

## Related API

[`fit_distance_distribution`](../reference/fitting.md#fastmob.measures.fitting.distance_distribution.fit_distance_distribution) ·
[`compare_distance_distributions`](../reference/fitting.md#fastmob.measures.fitting.distance_distribution.compare_distance_distributions) ·
[`DistributionFit`](../reference/fitting.md#fastmob.measures.fitting.distance_distribution.DistributionFit) ·
[`Gravity.fit`](../reference/models.md#fastmob.models.gravity.Gravity.fit) ·
[`compare_flow_models`](../reference/measures/evaluation.md#fastmob.measures.evaluation.compare_flow_models)
