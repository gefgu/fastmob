# How the fits work

**Question.** How does Fastkit-Mobility estimate a distance law or a gravity model, how sure can I be of the result, and what must I report so that a reader can check it?

This page explains the estimators, their limits and a reporting checklist. To run them, open [Compare to mobility laws](compare-to-mobility-laws.md) for distributions and [Validate against census flows](validate-against-census-flows.md) for flows.

## Distributions of distances

`fit_distance_distribution` fits one family to positive values (jump lengths, trip lengths, radii of gyration) by maximum likelihood. `compare_distance_distributions` fits several and ranks them.

| Step | What is done, and why it matters |
| --- | --- |
| Range | Values outside `[xmin, xmax]`, and zeros, are left out and counted in `n_excluded`. Default: smallest positive value to largest. Estimates depend on the range. Zeros are common (repeated check-ins at one place) and have no place in a power law. Raise `xmin` to ignore GPS noise. |
| Grouping | The values are grouped into 200 geometric bins on the range, and the likelihood is computed from the bin counts, so a million values cost no more than a thousand. |
| Likelihood | Each family is normalised on the same range, and the multinomial log-likelihood of the bin counts is maximised. Families are comparable: they explain the same data on the same range. |
| Uncertainty | The bin counts are resampled (bootstrap, 100 replicates by default) and the model refitted; the report has the standard error and the percentile interval of each parameter. Values from the same person are not independent: pass `groups=` (for example the user of each jump) to resample whole people, which gives wider and more honest intervals. |
| Fit quality | The KS distance between the empirical and the fitted cumulative distribution; with `n_gof` above 0, a parametric-bootstrap p-value. See the caveat below. |
| Comparison | AIC and BIC, their difference to the best model, and the Akaike weight. |

The five families are the power law `x^-beta`, the shifted power law `(x + r0)^-beta`, the exponential `exp(-x / kappa)`, the log-normal, and the truncated power law `(x + r0)^-beta exp(-x / kappa)` of González et al. (2008). The shifted and the truncated power law differ only by the cutoff `kappa`, so comparing them shows whether the data support a cutoff at all.

### Reading the result

- **A parameter at its bound.** Each parameter is searched within a range. If an estimate lands on the edge, `at_bound` lists it and the data do not identify it. For the truncated power law, a `kappa` at its upper bound (ten times the largest value by default) means there is no cutoff in the data: the model is a shifted power law, and a fitted `kappa` of 10^8 km is not evidence for a cutoff. Its standard error and interval are reported as NaN: say that the cutoff is not identified.
- **Independence.** The intervals and the p-value assume that the values are independent. Jumps from the same person are not, so intervals computed without `groups=` understate the uncertainty about people in general (they describe the fitting error only).
- **Goodness of fit.** With thousands of observations every simple model is rejected, because the test detects tiny departures. Report the KS distance and the AIC difference, and look at the fit on the data, instead of reading the p-value alone. A small p-value does not mean the model is useless, and a large one does not prove it right.
- **Several families.** An AIC difference below about 2 means the data cannot choose. The Akaike weight is a relative support among the families you list, not a probability that the family is true. The likelihood is on binned counts: use differences between families, not the absolute value.
- **Which estimate to report.** The histogram grid search and the maximum-likelihood fit give different exponents on the same data (for the check-in jumps 1.61 against 1.76, because their ranges and estimators differ). Report the maximum-likelihood one, with its range and interval.
- **The earlier fit.** `fit_values_to_truncated_powerlaw` fits a log histogram by a grid search and returns no uncertainty. It is kept for the plotting helpers; use `fit_distance_distribution` for a result that you report.

## Gravity models

`Gravity.fit` fits `flow ∝ origin^a * destination^b * f(distance)` by Poisson regression with a log link (Flowerdew and Murray, 1982), the standard estimator for flow counts (Santos Silva and Tenreyro, 2006, discuss its use for gravity models).

| Output | Meaning |
| --- | --- |
| `fit_summary()` | Estimate, standard error, 95% confidence interval and p-value of each parameter. The distance row is `distance_coefficient`, the exponent in `flow ∝ distance^gamma` (negative), or `distance_decay` for the exponential form, in km⁻¹. |
| `fit_statistics()` | Number of flows, log-likelihood, deviance, pseudo R-squared (one minus the deviance over the null deviance) and the dispersion. |

- **Robust errors.** Flow data are almost always overdispersed: their variance is larger than their mean, so the standard errors of a Poisson model are too small. The dispersion in `fit_statistics()` (Pearson chi-squared per degree of freedom) is 1 for a good Poisson fit and well above 1 for real flows (about 16 for the check-in flows in the notebook). On a table of expanded sample counts it also reflects the size of the weights, so do not read it as behavioural overdispersion there. `Gravity.fit` therefore reports heteroskedasticity-robust (sandwich) errors by default; the estimates do not change. Pass `robust=False` for the model-based errors.
- **Singly or globally constrained.** The singly constrained model has one free level for each origin, so it matches the outflow of every origin and the distance exponent is estimated from how flows spread among destinations. It is the right choice when the origin totals are observed (a census, or the origins of a sample). The globally constrained model has a single scale and fits only if the total flow of each origin follows the model.
- **Zeros.** A table of observed flows (such as the output of `trajectory_to_od`) lists only the pairs that have flow. Fitting those alone leaves out the long pairs that nobody travelled, and flattens the distance decay (for the check-in flows in the notebook, -1.21 without them and -1.40 with them). `Gravity.fit` therefore adds every missing pair of the catalogue as a flow of 0 (`include_zeros=True`), for origins that have at least one flow. This is the point of the Poisson estimator that Santos Silva and Tenreyro (2006) recommend.
- **Unweighted samples.** A bias that depends only on the origin (more people sampled in some home zones) does not bias the distance exponent of a singly constrained fit when the model holds, but the weights do change the estimate and its precision in a finite sample, and they change the flow levels. Weight the sample before comparing absolute flows (see the notebook).
- **Pseudo R-squared.** It compares the deviance with that of a model with only a constant. In the singly constrained model most of the explained deviance is the size of the origins, so a high value does not show that the distance law holds: look at the distance coefficient and its interval.
- **Relevance.** If the relevance of a destination is made of the same flows (for example check-ins counted at the destination), the destination exponent is partly circular. Prefer an independent size such as population or jobs.
- **Zones.** The exponent depends on the zone size and on the range of distances in the data. Report both.

## Comparing flows

`compare_flow_models` scores model flows against observed flows over the union of their origin-destination pairs.

| Score | Meaning |
| --- | --- |
| `cpc` | Common part of commuters: `2 * sum(min(observed, model)) / (sum(observed) + sum(model))`. 1 is identical. |
| `cpl` | Common part of links: the same for the sets of pairs that have a flow, whatever its size. |
| `r_squared`, `spearman` | Squared Pearson and rank correlation of the pair flows. |
| `poisson_deviance`, `pseudo_r_squared` | Fit of the model flows as the mean of Poisson counts, against a model that predicts the average flow everywhere. |

All models are scored on the same set of pairs (the union of all of them and the observed table), so the rows can be compared. All scores depend on the zone size. CPC is dominated by the largest flows, so a high CPC can hide a poor fit to the many small ones: look at the rank correlation too. CPL is only informative for integer (sampled) flows: expected flows are positive on every pair, which fixes their CPL. A model that was fitted to the observed flows is scored in-sample, so compare it with a parameter-free model (radiation), or score it on flows that were held out, as the notebook does.

## What to report

1. The data, the filtering, and the number of values or flows used.
2. For distributions: the family, the range `[xmin, xmax]` and `n_excluded`, the estimates with their intervals, the number of bootstrap replicates and whether people were resampled (`groups=`), the parameters at a bound, the KS distance (and the p-value with its replicates), and the comparison with the alternatives (AIC and weights).
3. For gravity fits: the deterrence function, the constraint type, whether zero pairs were included, the estimates with intervals, whether the errors are robust, the dispersion, the zone size and the source of the relevance.
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
