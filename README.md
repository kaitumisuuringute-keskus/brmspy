# brmspy

Python-first access to R's [brms](https://paul-buerkner.github.io/brms/)  with proper parameter names, ArviZ support, and cmdstanr performance. The easiest way to run brms models from Python.

This is an early development version of the library, use with caution.

[Github repo and issues](https://github.com/kaitumisuuringute-keskus/brmspy)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://kaitumisuuringute-keskus.github.io/brmspy/)
[![Coverage](https://kaitumisuuringute-keskus.github.io/brmspy/badges/coverage.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions)
[![python-test-matrix](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/python-test-matrix.yml/badge.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/python-test-matrix.yml)
[![r-dependencies-tests](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/r-dependencies-tests.yml/badge.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/r-dependencies-tests.yml)

## Installation

```bash
pip install brmspy
```

First-time setup (installs brms, cmdstanr, and CmdStan in R):

```python
from brmspy import brms
brms.install_brms() # requires R to be installed already
```

## Prebuilt Runtimes (Optional)

For faster installation (~20-60 seconds vs 20-30 minutes), use prebuilt runtime bundles:

```python
from brmspy import brms
brms.install_brms(use_prebuilt_binaries=True)
```

## Windows RTools

In case you don't have RTools installed, you can use the flag install_rtools = True. This is
disabled by default, because the flag runs the full rtools installer and modifies system path. 
Use with caution!

```python
from brmspy import brms
brms.install_brms(
    use_prebuilt_binaries=True,
    install_rtools=True # works for both prebuilt and compiled binaries.
)
```

### System Requirements

R >= 4.0

**Linux (x86_64):**
- glibc >= 2.27 (Ubuntu 18.04+, Debian 10+, RHEL 8+)
- g++ >= 9.0

**macOS (Intel & Apple Silicon):**
- Xcode Command Line Tools: `xcode-select --install`
- clang >= 11.0

**Windows (x86_64):**
- Rtools 4.0+ with MinGW toolchain
- g++ >= 9.0

Download Rtools from: https://cran.r-project.org/bin/windows/Rtools/

## Quick Start

```python
from brmspy import brms, prior
import arviz as az

# Load data
epilepsy = brms.get_brms_data("epilepsy")

# Fit model
model = brms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    priors=[
        prior("normal(0, 1)", "b"),
        prior("exponential(1)", "sd", group="patient"),
        prior("student_t(3, 0, 2.5)", "Intercept")
    ],
    chains=4,
    iter=2000
)

# Analyze
az.summary(model.idata)
az.plot_posterior(model.idata)
```

## Key Features

- **Proper parameter names**: Returns `b_Intercept`, `b_zAge`, `sd_patient__Intercept` instead of generic names like `b_dim_0`
- **arviz integration**: Returns `arviz.InferenceData` by default for Python workflow
- **brms formula syntax**: Full support for brms formula interface including random effects
- **Dual access**: Results include both `.idata` (arviz) and `.r` (brmsfit) attributes
- **No reimplementation**: Delegates all modeling logic to real brms. No Python-side reimplementation, no divergence from native behavior. Opinionated wrappers that rebuild formulas or stancode in Python inevitably drift from brms and accumulate their own bugs.
- **Prebuilt Binaries**: Fast installation with precompiled runtimes containing cmdstanr and brms (50x faster, 25 seconds on Google Colab)

## API Reference

[brmspy documentation](https://kaitumisuuringute-keskus.github.io/brmspy/)
[brms documentation](https://paulbuerkner.com/brms/reference/index.html)

### Setup Functions
- `brms.install_brms()` - Install brms, cmdstanr, and CmdStan
- `brms.get_brms_version()` - Get installed brms version

### Data Functions
- `brms.get_brms_data()` - Load example datasets from brms
- `brms.save_rds()` - Save brmsfit or another robject
- `brms.load_rds_fit()` - Load saved brmsfit object as FitResult (with idata)
- `brms.load_rds_raw()` - Load r object


### Model Functions
- `brms.formula()` - Define formula with kwargs
- `brms.fit()` or `brms.brm()` - Fit Bayesian regression model
- `brms.summary()` - Generate summary statistics as DataFrame
- `brms.make_stancode()` - Generate Stan code for model

### Prior functions
- `brms.prior()` - Define a prior with same syntax as r-s `prior_string`
- `brms.get_prior()` - Get pd.DataFrame describing default priors
- `brms.default_prior()` - Get pd.DataFrame describing default priors

### Families functions
- `brms.family()` - Get family object of FitResult
- `brms.brmsfamily()` - Construct family object from kwargs
- `brms.families.gaussian()`, `...bernoulli()`, `...beta_binomial()`, etc - Wrappers around brmsfamily for faster family object construction 

### Prediction Functions
- `brms.posterior_epred()` - Expected value predictions (without noise)
- `brms.posterior_predict()` - Posterior predictive samples (with noise)
- `brms.posterior_linpred()` - Linear predictor values
- `brms.log_lik()` - Log-likelihood values


## Usage

### Basic Model

```python
from brmspy import brms

kidney = brms.get_brms_data("kidney")

model = brms.fit(
    formula="time ~ age + disease",
    data=kidney,
    family="gaussian",
    chains=4,
    iter=2000
)
```

### With Priors

```python
from brmspy import prior

model = brms.fit(
    formula="count ~ zAge + (1|patient)",
    data=epilepsy,
    family="poisson",
    priors=[
        prior("normal(0, 0.5)", "b"),
        prior("cauchy(0, 1)", "sd")
    ],
    chains=4
)
```

### Model Summary

```python
from brmspy import summary

# Get summary statistics as DataFrame
summary_df = summary(model)
print(summary_df)
```

### Predictions

```python
# Expected value (without noise)
epred = brms.posterior_epred(model, newdata=new_data)

# Posterior predictive (with noise)
ypred = brms.posterior_predict(model, newdata=new_data)

# Linear predictor
linpred = brms.posterior_linpred(model, newdata=new_data)

# Log likelihood
loglik = brms.log_lik(model, newdata=new_data)
```

### Access Both Python and R Objects

```python
model = brms.fit(formula="y ~ x", data=data, chains=4)

# Python workflow with arviz
az.summary(model.idata)
az.plot_trace(model.idata)

# R workflow (if needed)
import rpy2.robjects as ro
ro.r('summary')(model.r)
```

## Sampling Parameters

```python
model = brms.fit(
    formula="y ~ x + (1|group)",
    data=data,
    iter=2000,      # Total iterations per chain
    warmup=1000,    # Warmup iterations
    chains=4,       # Number of chains
    cores=4,        # Parallel cores
    thin=1,         # Thinning
    seed=123        # Random seed
)
```

## Requirements

**Python**: 3.10-3.14

**R packages** (auto-installed via `brms.install_brms()`):
- brms >= 2.20.0
- cmdstanr
- posterior

**Python dependencies**:
- rpy2 >= 3.5.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- arviz (optional, for InferenceData)

## Development

```bash
git clone https://github.com/kaitumisuuringute-keskus/brmspy.git
cd brmspy
./init-venv.sh
pytest tests/ -v
```

## Architecture

brmspy uses:
- **brms::brm()** with cmdstanr backend for fitting (ensures proper parameter naming)
- **posterior** R package for conversion to draws format
- **arviz** for Python-native analysis and visualization
- **rpy2** for Python-R communication

Previous versions used CmdStanPy directly, which resulted in generic parameter names. Current version calls brms directly to preserve brms' parameter renaming logic.

## License

Apache License 2.0

## Credits

- Original concept: [Adam Haber](https://github.com/adamhaber)
- Current maintainer: [Remi Sebastian Kits](https://github.com/braffolk)
- Built on [brms](https://paul-buerkner.github.io/brms/) by Paul-Christian Bürkner
