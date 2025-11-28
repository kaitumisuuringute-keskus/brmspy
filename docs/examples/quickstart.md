# Quick Start Guide

## Installation

```bash
pip install brmspy
```

## First Time Setup

Install brms and CmdStan:

```python
from brmspy import brms
brms.install_brms()
```

## Basic Model

```python
from brmspy import brms
import arviz as az

# Load example data
epilepsy = brms.get_brms_data("epilepsy")

# Fit model
model = brms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    chains=4,
    iter=2000
)

# Analyze with arviz
az.summary(model.idata)
az.plot_posterior(model.idata)
```

## With Priors

```python
from brmspy import prior

model = brms.fit(
    formula="count ~ zAge + (1|patient)",
    data=epilepsy,
    family="poisson",
    priors=[
        prior("normal(0, 0.5)", class_="b"),
        prior("cauchy(0, 1)", class_="sd")
    ],
    chains=4
)
```

## Model Summary

```python
from brmspy import summary

# Get summary as DataFrame
summary_df = summary(model)
print(summary_df)
```

## Predictions

```python
import pandas as pd

# New data for predictions
new_data = pd.DataFrame({
    'zAge': [0, 0.5, 1.0],
    'zBase': [0, 0, 0],
    'Trt': [0, 0, 0],
    'patient': [1, 1, 1]
})

# Expected value predictions
epred = brms.posterior_epred(model, newdata=new_data)

# Posterior predictive samples
ypred = brms.posterior_predict(model, newdata=new_data)
```

## Access R Object

If you need direct R functionality:

```python
import rpy2.robjects as ro

# Access R brmsfit object
ro.r('summary')(model.r)
ro.r('plot')(model.r)