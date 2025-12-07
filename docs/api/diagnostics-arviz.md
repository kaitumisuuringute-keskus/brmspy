# Diagnostics with ArviZ

This guide covers how to use ArviZ for comprehensive diagnostics with brmspy models. All fitted models return `arviz.InferenceData` objects by default, enabling seamless integration with ArviZ's extensive diagnostic toolkit.

**Key Feature**: brmspy's InferenceData outputs are in the correct format for both univariate and multivariate models, so any ArviZ analysis function works directly without additional conversion or configuration.

## InferenceData Structure

Each fitted model's `.idata` attribute contains:

- **posterior**: Parameter samples (population-level effects, group-level effects, etc.)
  All parameters retain brms naming conventions (e.g., `b_Intercept`, `b_zAge`, `sd_patient__Intercept`)
- **posterior_predictive**: Posterior predictive samples for each response variable
- **log_likelihood**: Pointwise log-likelihood values for model comparison (LOO, WAIC)
- **observed_data**: Original response variable values
- **coords**: Coordinate labels (chain, draw, obs_id) for indexing

## Basic Diagnostics with ArviZ

### Summary Statistics

Use [`az.summary()`](https://arviz-devs.github.io/arviz/api/generated/arviz.summary.html) to get posterior estimates with convergence diagnostics:

```python
import brmspy
import arviz as az

# Fit model
model = brmspy.fit("count ~ zAge + (1|patient)", data=data, family="poisson")

# Get summary with Rhat and ESS
summary = az.summary(model.idata)
print(summary)
#                mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
# b_Intercept   1.234  0.123   1.012    1.456      0.002    0.001    3421.0    3012.0   1.00
# b_zAge        0.567  0.089   0.398    0.732      0.001    0.001    4123.0    3456.0   1.00
# ...
```

### Convergence Diagnostics

Check for convergence issues:

```python
# Rhat values (should be < 1.01)
rhat = az.rhat(model.idata)
print(f"Max Rhat: {rhat.max().values}")

# Effective sample size
ess_bulk = az.ess(model.idata, method="bulk")
ess_tail = az.ess(model.idata, method="tail")
print(f"Min bulk ESS: {ess_bulk.min().values}")
```

### Trace Plots

Visualize MCMC chains:

```python
# All parameters
az.plot_trace(model.idata)

# Specific parameters only
az.plot_trace(model.idata, var_names=["b_Intercept", "b_zAge"])
```

## Posterior Predictive Checks

### Univariate Models

Use [`az.plot_ppc()`](https://arviz-devs.github.io/arviz/api/generated/arviz.plot_ppc.html) to assess model fit:

```python
# Basic posterior predictive check
az.plot_ppc(model.idata)

# With specific number of samples
az.plot_ppc(model.idata, num_pp_samples=100)

# Different plot types
az.plot_ppc(model.idata, kind="cumulative")
az.plot_ppc(model.idata, kind="scatter")
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
az.plot_ppc(mv_model.idata, var_names=["tarsus"])
az.plot_ppc(mv_model.idata, var_names=["back"])
```

## Model Comparison

### Leave-One-Out Cross-Validation (LOO)

Compute LOO information criterion for model comparison:

```python
# Univariate model
loo_result = az.loo(model.idata)
print(loo_result)
# Computed from 4000 posterior samples and 100 observations log-likelihood matrix.
#          Estimate       SE
# elpd_loo   -234.5      8.2
# p_loo         12.3      1.1
# looic        469.0     16.4

# Multivariate model - specify response variable
loo_tarsus = az.loo(mv_model.idata, var_name="tarsus")
loo_back = az.loo(mv_model.idata, var_name="back")
```

### WAIC (Widely Applicable Information Criterion)

Alternative to LOO for model comparison:

```python
waic_result = az.waic(model.idata)
print(waic_result)

# For multivariate models
waic_tarsus = az.waic(mv_model.idata, var_name="tarsus")
```

### Comparing Multiple Models

Use [`az.compare()`](https://arviz-devs.github.io/arviz/api/generated/arviz.compare.html) to compare multiple models:

```python
# Fit competing models
model1 = brmspy.fit("y ~ x1", data=data)
model2 = brmspy.fit("y ~ x1 + x2", data=data)
model3 = brmspy.fit("y ~ x1 * x2", data=data)

# Compare with LOO
comparison = az.compare({
    "model1": model1.idata,
    "model2": model2.idata,
    "model3": model3.idata
}, ic="loo")

print(comparison)
#         rank  loo    p_loo   d_loo   weight    se   dse  warning  loo_scale
# model3     0 -234.5   12.3    0.0    0.72    8.2   0.0    False        log
# model2     1 -237.8   10.1    3.3    0.24    8.0   2.1    False        log
# model1     2 -245.2    8.9   10.7    0.04    7.8   4.5    False        log

# Visualize comparison
az.plot_compare(comparison)
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
comparison_back = az.compare(
    {"model1": mv_model1.idata, "model2": mv_model2.idata},
    ic="loo",
    var_name="back"
)
print(comparison_back)

# Compare for 'tarsus' response
comparison_tarsus = az.compare(
    {"model1": mv_model1.idata, "model2": mv_model2.idata},
    ic="loo",
    var_name="tarsus"
)
```

## Advanced Visualizations

### Posterior Distributions

Visualize parameter posteriors:

```python
# Forest plot
az.plot_forest(model.idata, var_names=["b"])

# Posterior densities
az.plot_posterior(model.idata, var_names=["b_Intercept", "b_zAge"])

# With reference values
az.plot_posterior(
    model.idata,
    var_names=["b_zAge"],
    ref_val=0,  # Add reference line at 0
    hdi_prob=0.95
)
```

### Pairwise Relationships

Examine parameter correlations:

```python
# Pair plot for selected parameters
az.plot_pair(
    model.idata,
    var_names=["b_Intercept", "b_zAge"],
    kind="hexbin"
)

# Include divergences (if any)
az.plot_pair(
    model.idata,
    var_names=["b"],
    divergences=True
)
```

### Energy Plots

Diagnose sampling issues:

```python
az.plot_energy(model.idata)
```

## Complete Diagnostic Workflow

Here's a complete example showing the full diagnostic workflow:

```python
from brmspy import brms
import arviz as az
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
print(az.summary(model.idata))
assert all(az.rhat(model.idata) < 1.01), "Convergence issues detected"

# 2. Visualize chains
az.plot_trace(model.idata, var_names=["b"])
plt.tight_layout()
plt.show()

# 3. Posterior predictive check
az.plot_ppc(model.idata, num_pp_samples=100)
plt.show()

# 4. Model comparison
loo = az.loo(model.idata)
print(f"LOO: {loo.loo:.1f} ± {loo.loo_se:.1f}")

# 5. Examine specific parameters
az.plot_posterior(
    model.idata,
    var_names=["b_zAge", "b_Trt"],
    ref_val=0
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

When working with multivariate models, remember to specify the `var_name` parameter in ArviZ functions that operate on response variables (e.g., `az.loo()`, `az.waic()`, `az.plot_ppc()`).

### Performance

For large models or datasets, LOO computation can be slow. Consider using `az.loo(..., pointwise=False)` or WAIC as alternatives.

## See Also

- [arviz.summary](https://arviz-devs.github.io/arviz/api/generated/arviz.summary.html) - Posterior summary statistics
- [arviz.loo](https://arviz-devs.github.io/arviz/api/generated/arviz.loo.html) - Leave-one-out cross-validation
- [arviz.waic](https://arviz-devs.github.io/arviz/api/generated/arviz.waic.html) - WAIC information criterion
- [arviz.compare](https://arviz-devs.github.io/arviz/api/generated/arviz.compare.html) - Compare multiple models
- [arviz.plot_ppc](https://arviz-devs.github.io/arviz/api/generated/arviz.plot_ppc.html) - Posterior predictive checks
- [arviz.plot_trace](https://arviz-devs.github.io/arviz/api/generated/arviz.plot_trace.html) - MCMC trace plots
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/api/index.html) - Complete ArviZ API reference