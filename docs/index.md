# brmspy

Python-first access to R's [brms](https://paul-buerkner.github.io/brms/)  with proper parameter names, ArviZ support, and cmdstanr performance. The easiest way to run brms models from Python.

## Key Features

- **Proper parameter names**: Returns `b_Intercept`, `b_zAge`, `sd_patient__Intercept` instead of generic names like `b_dim_0`
- **arviz integration**: Returns `arviz.InferenceData` by default for Python workflow
- **brms formula syntax**: Full support for brms formula interface including random effects
- **Dual access**: Results include both `.idata` (arviz) and `.r` (brmsfit or other) attributes
- **Prebuilt Binaries**: Fast installation with precompiled runtimes containing brms and cmstanr (50x faster, 25 seconds on Google Colab)

## Installation

```bash
pip install brmspy
```

First-time setup (installs brms, cmdstanr, and CmdStan in R):

```python
from brmspy import brms
brms.install_brms() # requires R to be installed already
```


### Prebuilt Runtimes (Optional)

For faster installation (~20-60 seconds vs 20-30 minutes), use prebuilt runtime bundles:

```python
from brmspy import brms
brms.install_brms(use_prebuilt_binaries=True)
```

#### System Requirements

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
    priors=[
        prior("normal(0, 1)", "b"),
        prior("exponential(1)", "sd", group="patient"),
        prior("student_t(3, 0, 2.5)", "Intercept")
    ],
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