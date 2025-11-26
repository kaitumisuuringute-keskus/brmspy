# brmspy

**Pythonic interface to R's brms for Bayesian regression modeling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

brmspy brings the power of [brms](https://paul-buerkner.github.io/brms/) (Bayesian Regression Models using Stan) to Python, providing proper parameter names and seamless integration with the Python Bayesian ecosystem.

## Quick Start

```bash
pip install brmspy
```

```python
import brmspy

# One-time setup: install brms and CmdStan
brmspy.install_brms()

# Load example data
epilepsy = brmspy.get_brms_data("epilepsy")

# Fit model - returns arviz InferenceData by default
model = brmspy.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    family="poisson",
    chains=4,
    iter=2000
)
idata = model.idata

# Analyze with arviz
import arviz as az
az.plot_posterior(idata)
az.summary(idata)
```

## Key Features

- **Proper Parameter Names**: Returns `b_Intercept`, `b_zAge`, `sd_patient__Intercept` (not `b_dim_0`, `sd_1_dim_0`)
- **Pythonic by Default**: Returns `arviz.InferenceData` for seamless Python integration
- **Flexible**: Optional R `brmsfit` return for full brms functionality
- **Formula Syntax**: Use brms' intuitive formula interface
- **Modern Stack**: Python 3.8-3.14, brms + cmdstanr backend

## Installation

```bash
# Install from PyPI
pip install brmspy

# Install with optional dependencies
pip install brmspy[viz]    # includes arviz, matplotlib
pip install brmspy[all]    # includes all optional dependencies

# First-time setup (installs brms and CmdStan in R)
python -c "import brmspy; brmspy.install_brms()"
```

**Note:** The package is imported as `brmspy` but installed as `brmspy` from PyPI.

## Usage

### Basic Model

```python
import brmspy
import arviz as az

# Load data
kidney = brmspy.get_brms_data("kidney")

# Fit Gaussian model
model = brmspy.fit(
    formula="time ~ age + disease",
    data=kidney,
    family="gaussian"
)

# View summary
az.summary(model.idata)
```

### With Priors

```python
model = brmspy.fit(
    formula="count ~ zAge + (1|patient)",
    data=epilepsy,
    family="poisson",
    priors=[
        ("normal(0, 0.5)", "b"),
        ("cauchy(0, 1)", "sd")
    ]
)
```

### Sampling Parameters

```python
model = brmspy.fit(
    formula="y ~ x",
    data=data,
    iter=2000,      # Total iterations per chain
    warmup=1000,    # Warmup iterations
    chains=4,       # Number of chains
    cores=4,        # Parallel cores
    seed=123        # Reproducibility
)
```

## Requirements

**Python**: 3.10+

**R Packages** (auto-installed):
- brms ≥ 2.20.0
- cmdstanr
- posterior

**Python Dependencies**:
- rpy2 ≥ 3.5.0
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- arviz (optional, for InferenceData conversion)

## Development

```bash
# Clone repository
git clone https://github.com/kaitumisuuringute-keskus/brmspy.git
cd brmspy

# Setup environment (requires Python 3.10+)
./init-venv.sh

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=brmspy --cov-report=html
```

## License

Apache License 2.0

## Credits

- Original concept: [Adam Haber](https://github.com/adamhaber)
- v0.1.0 modernization: [Remi Sebastian Kits](https://github.com/braffolk)
- Powered by [brms](https://paul-buerkner.github.io/brms/) by Paul-Christian Bürkner
