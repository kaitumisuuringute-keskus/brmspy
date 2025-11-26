# Advanced Usage

## Custom Sampling Parameters

```python
from brmspy import brms

model = brms.fit(
    formula="y ~ x + (1|group)",
    data=data,
    iter=2000,      # Total iterations per chain
    warmup=1000,    # Warmup iterations
    chains=4,       # Number of chains
    cores=4,        # Parallel cores
    thin=1,         # Thinning
    seed=123,       # Random seed
    control={'adapt_delta': 0.95}  # Stan control parameters
)
```

## Multiple Prediction Types

```python
# Expected value (without observation noise)
epred = brms.posterior_epred(model, newdata=new_data)

# Posterior predictive (with observation noise)
ypred = brms.posterior_predict(model, newdata=new_data)

# Linear predictor
linpred = brms.posterior_linpred(model, newdata=new_data)

# Log likelihood
loglik = brms.log_lik(model, newdata=new_data)
```

## Dual Python-R Workflow

```python
import arviz as az
import rpy2.robjects as ro

model = brms.fit(formula="count ~ zAge + (1|patient)", data=data, family="poisson")

# Python analysis
az.summary(model.idata)
az.plot_trace(model.idata)

# R analysis (if needed)
ro.r('summary')(model.r)
ro.r('plot')(model.r)
ro.r('loo')(model.r)  # Leave-one-out cross-validation
```

## Different Families

### Gaussian (default)

```python
model = brms.fit(
    formula="y ~ x",
    data=data,
    family="gaussian"
)
```

### Poisson (count data)

```python
model = brms.fit(
    formula="count ~ treatment + (1|subject)",
    data=data,
    family="poisson"
)
```

### Binomial (binary outcomes)

```python
model = brms.fit(
    formula="success | trials(n) ~ x",
    data=data,
    family="binomial"
)
```

### Student-t (robust to outliers)

```python
model = brms.fit(
    formula="y ~ x",
    data=data,
    family="student"
)
```

## Complex Random Effects

### Nested Random Effects

```python
model = brms.fit(
    formula="y ~ x + (1|country/region/city)",
    data=data
)
```

### Random Slopes

```python
model = brms.fit(
    formula="y ~ x + (x|subject)",
    data=data
)
```

### Correlated Random Effects

```python
model = brms.fit(
    formula="y ~ x + (x + z|subject)",
    data=data
)
```

## Working with Results

### Extract Posterior Samples

```python
# Get posterior as xarray Dataset
posterior = model.idata.posterior

# Extract specific parameter
b_intercept = posterior['b_Intercept'].values

# Extract all coefficients
coefficients = {
    var: posterior[var].values 
    for var in posterior.data_vars 
    if var.startswith('b_')
}
```

### Model Comparison

```python
import arviz as az

model1 = brms.fit(formula="y ~ x", data=data)
model2 = brms.fit(formula="y ~ x + z", data=data)

# Compare models
comparison = az.compare({
    'model1': model1.idata,
    'model2': model2.idata
})
print(comparison)
```

## Debugging

### Check Stan Code

```python
stan_code = brms.get_stan_code(
    formula="count ~ zAge + (1|patient)",
    data=data,
    priors=[],
    family="poisson"
)
print(stan_code)
```

### Compile Without Sampling

```python
# Just compile the model
model = brms.fit(
    formula="y ~ x",
    data=data,
    sample=False  # Don't sample, just compile
)