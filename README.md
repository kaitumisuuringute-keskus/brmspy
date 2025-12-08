# brmspy

Python-first access to R's [brms](https://paul-buerkner.github.io/brms/)  with proper parameter names, ArviZ support, and cmdstanr performance. The easiest way to run brms models from Python.

This is an early development version of the library, use with caution.

[Github repo and issues](https://github.com/kaitumisuuringute-keskus/brmspy)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://kaitumisuuringute-keskus.github.io/brmspy/)

[![Coverage main](https://kaitumisuuringute-keskus.github.io/brmspy/badges/coverage.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions)
[![Coverage r dependencies](https://kaitumisuuringute-keskus.github.io/brmspy/badges/coverage-rdeps.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions)
[![python-test-matrix](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/python-test-matrix.yml/badge.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/python-test-matrix.yml)
[![r-dependencies-tests](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/r-dependencies-tests.yml/badge.svg)](https://github.com/kaitumisuuringute-keskus/brmspy/actions/workflows/r-dependencies-tests.yml)

## Installation

### R Configuration

R>=4 is required before installing brmspy.

On linux and macos you may need to manually point your environment to the R installation.

Run the following in your terminal:

```bash
# Set R_HOME and add lib directory to LD_LIBRARY_PATH
export R_HOME=$(R RHOME)
export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH}"
```

### Python

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
brms.install_brms(use_prebuilt=True)
```

## Windows RTools

In case you don't have RTools installed, you can use the flag install_rtools = True. This is
disabled by default, because the flag runs the full rtools installer and modifies system path. 
Use with caution!

```python
from brmspy import brms
brms.install_brms(
    use_prebuilt=True,
    install_rtools=True # works for both prebuilt and compiled binaries.
)
```

### System Requirements

**Linux (x86_64):**
- glibc >= 2.27 (Ubuntu 18.04+, Debian 10+, RHEL 8+)
- g++ >= 9.0
- R >= 4.3

**macOS (Intel & Apple Silicon):**
- Xcode Command Line Tools: `xcode-select --install`
- clang >= 11.0
- R >= 4.2

**Windows (x86_64):**
- Rtools 4.0+ with MinGW toolchain
- g++ >= 9.0
- R >= 4.5

Download Rtools from: https://cran.r-project.org/bin/windows/Rtools/

## Key Features

- **Proper parameter names**: Returns `b_Intercept`, `b_zAge`, `sd_patient__Intercept` instead of generic names like `b_dim_0`
- **ArviZ integration**: Returns `arviz.InferenceData` by default for Python workflow
- **brms formula syntax**: Full support for brms formula interface including random effects
- **Dual access**: Results include both `.idata` (arviz) and `.r` (brmsfit) attributes
- **No reimplementation**: Delegates all modeling logic to real brms. No Python-side reimplementation, no divergence from native behavior
- **Prebuilt Binaries**: Fast installation with precompiled runtimes (50x faster, ~25 seconds on Google Colab)
- **Stays true to brms**: Function names, parameters, and returned objects are designed to be as close as possible to brms
- **Composable formula DSL**: Build multivariate, non-linear, and distributional formulas by simply adding components together, identical to brms

## Examples

### 1. Quick Start

Basic Bayesian regression with ArviZ diagnostics:

```python
from brmspy import brms
import arviz as az

# Fit Poisson model with random effects
epilepsy = brms.get_brms_data("epilepsy")
model = brms.brm("count ~ zAge + (1|patient)", data=epilepsy, family="poisson")

# Proper parameter names automatically!
print(az.summary(model.idata))
#                  mean     sd  hdi_3%  hdi_97%  ...  r_hat
# b_Intercept     1.234  0.123   1.012    1.456  ...   1.00
# b_zAge          0.567  0.089   0.398    0.732  ...   1.00
# sd_patient__... 0.345  0.067   0.223    0.467  ...   1.00
```

### 2. Multivariate Models (Python vs R)

Model multiple responses simultaneously with seamless ArviZ integration:

<table>
<tr><th>Python (brmspy)</th><th>R (brms)</th></tr>
<tr>
<td>

```python
from brmspy import brms, bf, set_rescor
import arviz as az

# Fit multivariate model
mv = brms.brm(
    bf("mvbind(tarsus, back) ~ sex + (1|p|fosternest)")
    + set_rescor(True),
    data=btdata
)

# ArviZ just works!
az.loo(mv.idata, var_name="tarsus")
az.loo(mv.idata, var_name="back")
az.plot_ppc(mv.idata, var_names=["tarsus"])
```

</td>
<td>

```r
library(brms)
library(loo)

# Fit multivariate model
fit <- brm(
  bf(mvbind(tarsus, back) ~ sex + (1|p|fosternest))
  + set_rescor(TRUE),
  data = BTdata
)

# Separate LOO for each response
loo_tarsus <- loo(fit, resp = "tarsus")
loo_back <- loo(fit, resp = "back")
```

</td>
</tr>
</table>

### 3. Distributional Regression

Model heteroscedasticity (variance depends on predictors):

```python
from brmspy import bf

# Model both mean AND variance
model = brms.brm(
    bf("reaction ~ days", sigma = "~ days"),  # sigma varies with days!
    data=sleep_data,
    family="gaussian"
)

# Extract distributional parameters
print(model.idata.posterior.data_vars)
# b_Intercept, b_days, b_sigma_Intercept, b_sigma_days, ...
```

### 4. Complete Diagnostic Workflow with ArviZ

Full model checking in ~10 lines:

```python
from brmspy import brms
import arviz as az

model = brms.brm("count ~ zAge * Trt + (1|patient)", data=epilepsy, family="poisson")

# Check convergence
assert az.rhat(model.idata).max() < 1.01, "Convergence issues!"
assert az.ess(model.idata).min() > 400, "Low effective sample size!"

# Posterior predictive check
az.plot_ppc(model.idata, num_pp_samples=100)

# Model comparison
model2 = brms.brm("count ~ zAge + Trt + (1|patient)", data=epilepsy, family="poisson")
comparison = az.compare({"interaction": model.idata, "additive": model2.idata})
print(comparison)
#              rank  loo    p_loo  d_loo  weight
# interaction     0 -456.2   12.3    0.0    0.89
# additive        1 -461.5   10.8    5.3    0.11
```

### 5. Advanced Formulas: Splines & Non-linear Effects

Smooth non-linear relationships with splines:

```python
from brmspy import brms

# Generalized additive model (GAM) with spline
model = brms.brm(
    "y ~ s(x, bs='cr', k=10) + (1 + x | group)",
    data=data,
    family="gaussian"
)

# Polynomial regression
poly_model = brms.brm(
    "y ~ poly(x, 3) + (1|group)",
    data=data
)

# Extract and visualize smooth effects
conditional_effects = brms.call("conditional_effects", model, "x")
```

### Additional Features

**Custom Priors:**
```python
from brmspy import prior

model = brms.brm(
    "count ~ zAge + (1|patient)",
    data=epilepsy,
    priors=[
        prior("normal(0, 0.5)", class_="b"),
        prior("exponential(1)", class_="sd", group="patient")
    ],
    family="poisson"
)
```

**Predictions:**
```python
import pandas as pd

new_data = pd.DataFrame({"zAge": [-1, 0, 1], "patient": [999, 999, 999]})

# Expected value (without observation noise)
epred = brms.posterior_epred(model, newdata=new_data)

# Posterior predictive (with noise)
ypred = brms.posterior_predict(model, newdata=new_data)

# Access as InferenceData for ArviZ
az.plot_violin(epred.idata)
```

### 6. Maximalist Example: Kitchen Sink

Everything at once - multivariate responses, different families, distributional parameters, splines, and complete diagnostics:

```python
from brmspy import brms, bf, lf, set_rescor, skew_normal, gaussian
import arviz as az

# Load data
btdata = brms.get_data("BTdata", package="MCMCglmm")

bf_tarsus = (
    bf("tarsus ~ sex + (1|p|fosternest) + (1|q|dam)") +
    lf("sigma ~ 0 + sex") +
    skew_normal()
)

bf_back = (
    bf("back ~ s(hatchdate) + (1|p|fosternest) + (1|q|dam)") +
    gaussian()
)

model = brms.brm(
    bf_tarsus + bf_back + set_rescor(False),
    data=btdata,
    chains=2,
    control={"adapt_delta": 0.95}
)

# ArviZ diagnostics work seamlessly
for response in ["tarsus", "back"]:
    print(f"\n=== {response.upper()} ===")
    
    # Model comparison
    loo = az.loo(model.idata, var_name=response)
    print(f"LOO: {loo.loo:.1f} ± {loo.loo_se:.1f}")
    
    # Posterior predictive check
    az.plot_ppc(model.idata, var_names=[response])
    
    # Parameter summaries
    print(az.summary(
        model.idata,
        var_names=[f"b_{response}"],
        filter_vars="like"
    ))

# Visualize non-linear effect
conditional = brms.call("conditional_effects", model, "hatchdate", resp="back")
# Returns proper pandas DataFrame ready for plotting!
```

**Output shows:**
- Proper parameter naming: `b_tarsus_Intercept`, `b_tarsus_sex`, `b_sigma_sex`, `sd_fosternest__tarsus_Intercept`, etc.
- Separate posterior predictive for each response
- Per-response LOO for model comparison
- All parameters accessible via ArviZ


## API Reference (partial)

[brmspy documentation](https://kaitumisuuringute-keskus.github.io/brmspy/)

[brms documentation](https://paulbuerkner.com/brms/reference/index.html)

### Setup Functions
It is NOT recommended to run installation functions when you have used the session.

- `install_brms()` - Install brms, cmdstanr, and CmdStan from source or runtime
- `install_runtime()` - Install latest runtime for OS
- `activate_runtime()` - Activate existing prebuilt runtime
- `deactivate_runtime()` - Deactivate current runtime - May break on windows.
- `get_brms_version()` - Get installed brms version
- `find_local_runtime()` - checks if a runtime exists locally in standard directory and returns path if it does

### Data Functions
- `get_brms_data()` - Load example datasets from brms
- `get_data()` - Load example datasets from any package
- `save_rds()` - Save brmsfit or another robject
- `load_rds_fit()` - Load saved brmsfit object as FitResult (with idata)
- `load_rds_raw()` - Load r object

### Model Functions
- `bf`, `lg`, `nlf`, `acformula`, `set_rescor`, `set_mecor`, `set_nl` - formula functions
- `brm()` - Fit Bayesian regression model
- `add_criterion` - add loo, waic criterions to fit
- `make_stancode()` - Generate Stan code for model

### Diagnostics Functions
- `summary()` - Comprehensive model summary as SummaryResult dataclass
- `fixef()` - Extract population-level (fixed) effects
- `ranef()` - Extract group-level (random) effects as xarray
- `posterior_summary()` - Summary statistics for all parameters
- `prior_summary()` - Extract prior specifications used in model
- `validate_newdata()` - Validate new data for predictions
- For loo, waic etc use arviz!

### Prior Functions
- `prior()` - Define a prior with same syntax as r-s `prior_string`
- `get_prior()` - Get pd.DataFrame describing default priors
- `default_prior()` - Get pd.DataFrame describing default priors

### Families Functions
- `family()` - Get family object of FitResult
- `brmsfamily()` - Construct family object from kwargs
- `gaussian()`, `...bernoulli()`, `...beta_binomial()`, etc - Wrappers around brmsfamily for faster family object construction

### Prediction Functions
- `posterior_epred()` - Expected value predictions (without noise)
- `posterior_predict()` - Posterior predictive samples (with noise)
- `posterior_linpred()` - Linear predictor values
- `log_lik()` - Log-likelihood values

### Generic Function Access
- `call()` - Call any brms/R function by name with automatic type conversion


## Known issues

- Due to Windows' idiosyncrasies installing existing R packages (or cmdstanr) is NOT guaranteed to succeed in the same session if it has already been used. It is strongly recommended to restart your Python session before doing any installations when you have used it. This also means autoloading previously used prebuilt environment on windows is disabled, call activate() to load existing prebuilt runtime.

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

- Current maintainer: [Remi Sebastian Kits](https://github.com/braffolk)
- Original concept: [Adam Haber](https://github.com/adamhaber)
- Built on [brms](https://paul-buerkner.github.io/brms/) by Paul-Christian Bürkner
