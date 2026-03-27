# Diagnostics with ArviZ

This guide covers how to use ArviZ for comprehensive diagnostics with brmspy models. All fitted models return `ArviZ DataTree / InferenceData` objects by default, enabling seamless integration with ArviZ's extensive diagnostic toolkit.

**Key Feature**: brmspy's ArviZ data structure outputs are correctly formatted for both univariate and multivariate models, so any ArviZ analysis function works directly without additional conversion or configuration.

Note: The documentation reflects usage of ArviZ 1.0 API.

## Data Structure

Each fitted model's `.idata` attribute contains:

- **posterior**: Parameter samples (population-level effects, group-level effects, etc.)
  All parameters retain brms naming conventions (e.g., `b_Intercept`, `b_zAge`, `sd_patient__Intercept`)
- **posterior_predictive**: Posterior predictive samples for each response variable
- **log_likelihood**: Pointwise log-likelihood values for model comparison (LOO, WAIC)
- **observed_data**: Original response variable values
- **coords**: Coordinate labels (chain, draw, obs_id) for indexing

## Basic Diagnostics with ArviZ

### Summary Statistics

Use `arviz_stats.summary()` to get posterior estimates with convergence diagnostics:

```python
import brmspy
from arviz_stats import summary

# Fit model
model = brmspy.fit("count ~ zAge + (1|patient)", data=data, family="poisson")

# Get summary with Rhat and ESS
summ = summary(model.idata)
print(summ)
#                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# b_Intercept   1.234  0.123   1.012    1.456      0.002    0.001    3421.0    3012.0   1.00
# b_zAge        0.567  0.089   0.398    0.732      0.001    0.001    4123.0    3456.0   1.00
# ...
```

### Convergence Diagnostics

Check for convergence issues:

```python
from arviz_stats import rhat, ess

# Rhat values (should be < 1.01)
rh = rhat(model.idata)
print(f"Max Rhat: {rh.max().values}")

# Effective sample size
ess_bulk = ess(model.idata, method="bulk")
ess_tail = ess(model.idata, method="tail")
print(f"Min bulk ESS: {ess_bulk.min().values}")
```

### Trace Plots

Visualize MCMC chains:

```python
from arviz_plots import plot_trace

# All parameters
plot_trace(model.idata)

# Specific parameters only
plot_trace(model.idata, var_names=["b_Intercept", "b_zAge"])
```

## Posterior Predictive Checks

### Univariate Models

Use `arviz_plots.plot_ppc_dist()` to assess model fit:

```python
from arviz_plots import plot_ppc_dist

# Basic posterior predictive check
plot_ppc_dist(model.idata)

# With specific number of samples
plot_ppc_dist(model.idata, num_samples=100)
```

### Multivariate Models

For multivariate models with multiple response variables, specify which response to check using the `var_names` parameter:

```python
# Fit multivariate model
from brmspy import bf, set_rescor

mv_model = brmspy.fit(
    bf("mvbind(tarsus, back) ~ sex + (1|p|fosternest)") + set_rescor(True),
    data=data
)

# Check each response separately
plot_ppc_dist(mv_model.idata, var_names=["tarsus"])
plot_ppc_dist(mv_model.idata, var_names=["back"])
```

## Model Comparison

### Leave-One-Out Cross-Validation (LOO)

Compute LOO information criterion for model comparison:

```python
from arviz_stats import loo

# Univariate model
loo_result = loo(model.idata)
print(loo_result)

# Multivariate model - specify response variable
loo_tarsus = loo(mv_model.idata, var_names=["tarsus"])
loo_back = loo(mv_model.idata, var_names=["back"])
```

### Comparing Multiple Models

Use `arviz_stats.compare()` to compare multiple models:

```python
from arviz_stats import compare
from arviz_plots import plot_compare

# Fit competing models
model1 = brmspy.fit("y ~ x1", data=data)
model2 = brmspy.fit("y ~ x1 + x2", data=data)
model3 = brmspy.fit("y ~ x1 * x2", data=data)

# Compare with LOO
comp = compare({
    "model1": model1.idata,
    "model2": model2.idata,
    "model3": model3.idata
})

print(comp)

# Visualize comparison
plot_compare(comp)
```

### Multivariate Model Comparison

For multivariate models, compare each response separately:

```python
# Fit two multivariate models
mv_model1 = brmspy.fit(
    bf("mvbind(tarsus, back) ~ sex + (1|p|fosternest)") + set_rescor(True),
    data=data
)

mv_model2 = brmspy.fit(
    bf("mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest)") + set_rescor(True),
    data=data
)

# Compare for 'back' response
comparison_back = compare(
    {"model1": mv_model1.idata, "model2": mv_model2.idata},
    var_names=["back"]
)
print(comparison_back)

# Compare for 'tarsus' response
comparison_tarsus = compare(
    {"model1": mv_model1.idata, "model2": mv_model2.idata},
    var_names=["tarsus"]
)
```

## Advanced Visualizations

### Posterior Distributions

Visualize parameter posteriors:

```python
from arviz_plots import plot_forest, plot_dist

# Forest plot
plot_forest(model.idata, var_names=["b"])

# Posterior densities
plot_dist(model.idata, var_names=["b_Intercept", "b_zAge"])

# With reference values (Note: parameter varies by plot type in 1.0)
plot_dist(
    model.idata,
    var_names=["b_zAge"]
)
```

### Pairwise Relationships

Examine parameter correlations:

```python
from arviz_plots import plot_pair

# Pair plot for selected parameters
plot_pair(
    model.idata,
    var_names=["b_Intercept", "b_zAge"],
    kind="hexbin"
)

# Include divergences (if any)
plot_pair(
    model.idata,
    var_names=["b"],
    divergences=True
)
```

### Energy Plots

Diagnose sampling issues:

```python
from arviz_plots import plot_energy
plot_energy(model.idata)
```

## Complete Diagnostic Workflow

Here's a complete example showing the full diagnostic workflow:

```python
from brmspy import brms
from arviz_stats import summary, rhat, loo
from arviz_plots import plot_trace, plot_ppc_dist, plot_dist
import matplotlib.pyplot as plt

# Fit model
epilepsy = brms.get_brms_data("epilepsy")
model = brms.fit(
    "count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    chains=4,
    iter=2000
)

# 1. Check convergence
print(summary(model.idata))
assert all(rhat(model.idata).to_array().values < 1.01), "Convergence issues detected"

# 2. Visualize chains
plot_trace(model.idata, var_names=["b"])
plt.tight_layout()
plt.show()

# 3. Posterior predictive check
plot_ppc_dist(model.idata, num_samples=100)
plt.show()

# 4. Model comparison
loo_res = loo(model.idata)
print(loo_res)

# 5. Examine specific parameters
plot_dist(
    model.idata,
    var_names=["b_zAge", "b_Trt"]
)
plt.show()
```

## Notes

### Parameter Naming

brmspy preserves brms parameter naming conventions:

- Population-level effects: `b_Intercept`, `b_variable_name`
- Group-level standard deviations: `sd_group__effect`
- Correlations: `cor_group__effect1__effect2`
- Family-specific parameters: `sigma`, `nu`, `shape`, etc.

### Multivariate Models

When working with multivariate models, remember to specify the `var_names` parameter in ArviZ functions that operate on response variables (e.g., `loo()`, `plot_ppc_dist()`).

### Performance

For large models or datasets, LOO computation can be slow. Consider using `az.loo(..., pointwise=False)` or WAIC as alternatives.

## See Also

- [arviz_stats API](https://python.arviz.org/projects/stats/en/stable/api/index.html) - Statistical and diagnostic functions
- [arviz_plots API](https://python.arviz.org/projects/plots/en/stable/api/index.html) - Plotting and visualization functions