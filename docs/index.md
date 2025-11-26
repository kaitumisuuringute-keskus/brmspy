# brmspy

Python-first access to R's brms with proper parameter names, ArviZ support, and cmdstanr performance. The easiest way to run brms models from Python.

## Overview

brmspy provides Python access to [brms](https://paul-buerkner.github.io/brms/) with proper parameter naming and seamless arviz integration. Uses brms with cmdstanr backend.

## Key Features

- **Proper parameter names**: Returns `b_Intercept`, `b_zAge`, `sd_patient__Intercept` instead of generic names like `b_dim_0`
- **arviz integration**: Returns `arviz.InferenceData` by default for Python workflow
- **brms formula syntax**: Full support for brms formula interface including random effects
- **Dual access**: Results include both `.idata` (arviz) and `.r` (brmsfit) attributes

## Installation

```bash
pip install brmspy
```

First-time setup (installs brms, cmdstanr, and CmdStan in R):

```python
from brmspy import brms
brms.install_brms()
```

## Quick Start

```python
from brmspy import brms
import arviz as az

# Load data
epilepsy = brms.get_brms_data("epilepsy")

# Fit model
model = brms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    chains=4,
    iter=2000
)

# Analyze
az.summary(model.idata)
az.plot_posterior(model.idata)
```

## Requirements

- Python 3.10-3.14
- R with brms >= 2.20.0, cmdstanr, and posterior packages

## Links

- [GitHub Repository](https://github.com/kaitumisuuringute-keskus/brmspy)
- [API Reference](api/brms.md)
- [Examples](examples/quickstart.md)