# Fit and evaluate mobility results

Use fitting APIs to summarize an observed distribution, then use evaluation
APIs to compare observations, simulations, or alternative preprocessing runs.

```python
from fastmob.measures.fitting import fit_values_to_truncated_powerlaw
from fastmob.measures.evaluation import wasserstein_distance

fit = fit_values_to_truncated_powerlaw(jump_values)
distance = wasserstein_distance(observed_values, simulated_values)
```

Choose the comparison at the same level as the question:

| Compare | Use |
| --- | --- |
| Scalar predictions | `mse`, `rmse`, `r_squared` |
| Ranked or continuous distributions | `wasserstein_distance`, divergence metrics |
| Origin-destination flows | `common_part_of_commuters`, `common_part_of_links` |
| Spatiotemporal visitation distributions | `stvd_emd` |

For dataframe-level grouped comparisons, use the typed dataframe
`compare_to()` method rather than constructing a manual join. See the
[fitting reference](../reference/fitting.md) and
[evaluation reference](../reference/measures/evaluation.md) for assumptions,
output fields, and optional dependencies.

<!-- Visual placeholder: observed and simulated distribution comparison. -->
