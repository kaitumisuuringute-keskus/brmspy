"""
brmspy - Python interface to R's brms for Bayesian regression modeling

Provides Pythonic access to the brms R package with proper parameter naming
and seamless arviz integration. Uses brms with cmdstanr backend for optimal
performance and Stan parameter naming.

Key Features
------------
- **Pythonic API**: Snake_case parameters, Python types, pandas DataFrames
- **ArviZ Integration**: Automatic conversion to InferenceData objects
- **Type Safety**: Full type hints and IDE autocomplete support
- **Flexible Priors**: Type-safe prior specification with `prior()` helper
- **Multiple Backends**: Support for both cmdstanr (recommended) and rstan
- **Prebuilt Binaries**: Fast installation with precompiled runtimes (50x faster)

Installation
------------
Basic installation (requires R >= 4.2 and Python >= 3.8):

```python
pip install brmspy
```

Install R dependencies (traditional method, ~30 minutes):

```python
    from brmspy import brms
    brms.install_brms()
```
Quick installation with prebuilt binaries (recommended, ~1 minute):

```python
    brms.install_brms(use_prebuilt=True)
```
Install specific brms version:

```python
    brms.install_brms(brms_version="2.23.0")
```
Quick Start
-----------
Basic Bayesian regression workflow:

```python
    from brmspy import brms
    import arviz as az

    # Load example data
    epilepsy = brms.get_brms_data("epilepsy")

    # Fit Poisson regression model
    model = brms.fit(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        chains=4,
        iter=2000
    )

    # Analyze results with ArviZ
    az.summary(model.idata)
    az.plot_trace(model.idata)
```

Using custom priors:

```
from brmspy import brms, prior

model = brms.fit(
    formula="count ~ zAge + zBase * Trt + (1|patient)",
    data=epilepsy,
    priors=[
        prior("normal(0, 0.5)", class_="b"),
        prior("exponential(2)", class_="sd", group="patient")
    ],
    family="poisson",
    chains=4
)
```
Making predictions:

```
# Posterior predictive (with observation noise)
preds = brms.posterior_predict(model, newdata=new_data)
az.plot_ppc(preds.idata)

# Expected values (without noise)
epred = brms.posterior_epred(model, newdata=new_data)

# Linear predictor
linpred = brms.posterior_linpred(model, newdata=new_data)
```

Model comparison and diagnostics:

```python
# Model comparison with LOO-CV
import arviz as az
loo = az.loo(model.idata)
print(loo)

# Convergence diagnostics
az.rhat(model.idata)
az.ess(model.idata)

# Posterior predictive checks
az.plot_ppc(model.idata)
```

See Also
--------
- Documentation: https://brmspy.readthedocs.io
- GitHub: https://github.com/kaitumisuuringute-keskus/brmspy
- brms R package: https://paulbuerkner.com/brms/
- ArviZ documentation: https://arviz-devs.github.io/arviz/

Notes
-----
brmspy requires:

- Python >= 3.10 (3.10-3.14 supported)
- R >= 4.2
- rpy2 for Python-R interface
- Working C++ compiler for Stan models (g++ >= 9, clang >= 11, or Rtools on Windows)

For optimal performance, use cmdstanr backend (default) over rstan. The cmdstanr
backend provides proper Stan parameter naming and faster compilation.

Examples
--------
Complete workflow with model diagnostics:

```python
    from brmspy import brms, prior
    import arviz as az
    import pandas as pd

    # Load data
    data = brms.get_brms_data("kidney")

    # Define model with informative priors
    model = brms.fit(
        formula="time | cens(censored) ~ age + sex + disease + (1 + age | patient)",
        data=data,
        family="weibull",
        priors=[
            prior("normal(0, 1)", class_="b"),
            prior("student_t(3, 0, 2.5)", class_="Intercept"),
            prior("exponential(1)", class_="sd"),
            prior("lkj(2)", class_="cor")
        ],
        chains=4,
        iter=4000,
        warmup=2000,
        cores=4,
        seed=42
    )

    # Check convergence
    print(az.summary(model.idata, var_names=["b"]))
    assert all(az.rhat(model.idata) < 1.01)

    # Visualize results
    az.plot_trace(model.idata, var_names=["b", "sd"])
    az.plot_posterior(model.idata, var_names=["b"])

    # Make predictions
    new_patients = pd.DataFrame({
        "age": [50, 60, 70],
        "sex": [0, 1, 0],
        "disease": [1, 1, 2],
        "patient": [999, 999, 999]
    })
    predictions = brms.posterior_predict(model, newdata=new_patients)
    print(predictions.idata.posterior_predictive)
```
"""

__version__ = "0.3.0"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"

from brmspy import brms

__all__ = ["brms"]
