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

"""
__version__ = "0.3.0"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"


from typing import List, Optional, cast, Tuple


def _check_r_setup(
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    import shutil, subprocess, os, platform

    ok = True
    messages: List[str] = []

    def info(msg: str) -> None:
        if verbose:
            print(f"[brmspy][INFO] {msg}")

    def warn(msg: str) -> None:
        nonlocal ok
        ok = False
        messages.append(msg)
        if verbose:
            print(f"[brmspy][WARNING] {msg}")

    # --- 1. Try to locate R and RHOME via the R executable -----------------
    r_exec = shutil.which("R")
    r_home_cmd: Optional[str] = None

    if not r_exec:
        # Not necessarily fatal if rpy2 was compiled with an absolute R_HOME,
        # but very suspicious for anything reproducible.
        warn("R executable `R` not found on PATH; this is a fragile setup.")
    else:
        try:
            proc = subprocess.run(
                [r_exec, "RHOME"],
                check=True,
                capture_output=True,
                text=True
            )
            r_home_cmd = proc.stdout.strip()
            if not r_home_cmd:
                warn("`R RHOME` returned an empty value.")
            else:
                info(f"RHOME (from `R RHOME`): {r_home_cmd}")
        except Exception as e:
            warn(f"`R RHOME` failed: {e!r}")

    # --- 2. Look at env vars, but treat them as advisory -------------------
    r_home_env = os.environ.get("R_HOME")
    if r_home_env:
        info(f"R_HOME env: {r_home_env}")
    else:
        info("R_HOME env var not set; relying on rpy2 / system defaults.")

    if platform.system() != "Windows":
        ld = os.environ.get("LD_LIBRARY_PATH")
        if not ld:
            info(
                "LD_LIBRARY_PATH not set; assuming system linker config "
                "already knows where libR.so lives."
            )
        else:
            info(f"LD_LIBRARY_PATH is set (length {len(ld)} chars).")

    # --- 3. Try to import rpy2 and talk to R --------------------------------
    try:
        import rpy2.robjects as ro  # type: ignore[import]
    except Exception as e:
        warn(f"Failed to import rpy2.robjects: {e!r}")
        return ok, messages

    try:
        r_version = str(cast(ro.ListVector, ro.r("R.version"))[0])
        lib_paths = [str(p) for p in cast(ro.ListVector, ro.r(".libPaths()"))]
        info(f"R version: {r_version}")
        info(f".libPaths(): {lib_paths}")
    except Exception as e:
        warn(f"rpy2 could not initialize R / run basic code: {e!r}")
        return ok, messages

    return ok, messages

def _initialise_r_safe() -> None:


    import os
    import sys

    # CFFI MODE
    if "rpy2" in sys.modules:
        if os.environ.get('RPY2_CFFI_MODE') != "ABI":
            print(
                "[brmspy][WARNING] rpy2 was imported before brmspy; cannot enforce "
                "RPY2_CFFI_MODE (env var). API and BOTH mode are known to cause "
                "instability, ABI is recommended."
            )
    elif os.environ.get('RPY2_CFFI_MODE') in ('BOTH', 'API'):
        print(
            "[brmspy][WARNING] RPY2_CFFI_MODE (env var) is set to API/BOTH. "
            "These modes are known to cause instability and segfaults; "
            "ABI is recommended."
        )
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    # THREAD SAFETY
    # Could also lead to undefined behaviour if >1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ok, messages = _check_r_setup()
    if not ok:
        print("[brmspy][WARNING] R environment diagnostics reported problems.")
        for message in messages:
            print(f"[brmspy][WARNING]   {message}")

    import rpy2.robjects as ro

    ro.r(
        r'''
        # Disable fork-based mechanisms that are unsafe in embedded R
        options(
          mc.cores = 1L,             # parallel::mclapply -> serial
          future.fork.enable = FALSE, # disable future::multicore
          loo.cores = 1L # deprecated but still respected, for now.
        )

        # If 'future' is installed, force sequential backend
        if (requireNamespace("future", quietly = TRUE)) {
          future::plan(future::sequential)
        }
        '''
    )

_initialise_r_safe()

# Import brms module for use as: from brmspy import brms
from brmspy import brms
from brmspy import _runtime
from brmspy._runtime import (
    install_brms, install_runtime,
    deactivate_runtime, activate_runtime,
    find_local_runtime, get_active_runtime,
    get_brms_version,
)
from brmspy._runtime._platform import system_fingerprint
from brmspy.brms import (
    get_brms_data,
    get_data,
    fit, brm,

    formula, bf, set_nl, set_mecor, set_rescor,
    lf, nlf, acformula,

    make_stancode,
    posterior_epred,
    posterior_predict,
    posterior_linpred,
    log_lik,
    summary, fixef, ranef,
    prior_summary, posterior_summary, validate_newdata,
    prior, get_prior, default_prior,

    call,

    save_rds, read_rds_fit, read_rds_raw,

    FitResult,
    PosteriorEpredResult,
    PosteriorPredictResult,
    PosteriorLinpredResult,
    LogLikResult,
    GenericResult,
    FormulaResult,
    IDLinpred,
    IDEpred,
    IDFit,
    IDLogLik,
    IDPredict,
    PriorSpec,

    install_rpackage
)
from brmspy._brms_functions.families import (
    brmsfamily, family, student, bernoulli, beta_binomial, negbinomial,
    negbinomial2, geometric, discrete_weibull, com_poisson, lognormal,
    shifted_lognormal, skew_normal, exponential, weibull, frechet,
    gen_extreme_value, exgaussian, wiener, Beta, xbeta, dirichlet,
    dirichlet2, logistic_normal, von_mises, asym_laplace,
    zero_inflated_asym_laplace, cox, hurdle_poisson, hurdle_negbinomial,
    hurdle_gamma, hurdle_lognormal, hurdle_cumulative, zero_inflated_beta,
    zero_one_inflated_beta, zero_inflated_poisson, zero_inflated_negbinomial,
    zero_inflated_binomial, zero_inflated_beta_binomial, categorical,
    multinomial, dirichlet_multinomial, cumulative, sratio, cratio, acat,
    gaussian, poisson, binomial, Gamma, inverse_gaussian
)
__all__ = [
    # R env
    'install_brms', 'install_runtime', 'get_brms_version',  'deactivate_runtime', 'activate_runtime',
    'find_local_runtime', 'get_active_runtime',

    # IO
    'get_brms_data', 'save_rds', 'read_rds_raw', 'read_rds_fit', 'get_data',

    # brm
    'fit', 'brm',

    # formula
    'formula', 'bf', 'set_mecor', 'set_rescor', 'set_nl',
    'lf', 'nlf', 'acformula',

    # priors
    'prior', 'get_prior', 'default_prior',

    # prediction
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",

    # families
    "brmsfamily", "family", "student", "bernoulli", "beta_binomial", "negbinomial",
    "negbinomial2", "geometric", "discrete_weibull", "com_poisson", "lognormal",
    "shifted_lognormal", "skew_normal", "exponential", "weibull", "frechet",
    "gen_extreme_value", "exgaussian", "wiener", "Beta", "xbeta", "dirichlet",
    "dirichlet2", "logistic_normal", "von_mises", "asym_laplace",
    "zero_inflated_asym_laplace", "cox", "hurdle_poisson", "hurdle_negbinomial",
    "hurdle_gamma", "hurdle_lognormal", "hurdle_cumulative", "zero_inflated_beta",
    "zero_one_inflated_beta", "zero_inflated_poisson", "zero_inflated_negbinomial",
    "zero_inflated_binomial", "zero_inflated_beta_binomial", "categorical",
    "multinomial", "dirichlet_multinomial", "cumulative", "sratio", "cratio", "acat",
    "gaussian", "poisson", "binomial", "Gamma", "inverse_gaussian",

    # diagnosis
    'summary', 'fixef', 'ranef', 'prior_summary', 'posterior_summary',
    'validate_newdata',

    # generic helper
    'call', 'install_rpackage',

    # types
    'FitResult', 'FormulaResult', 'PosteriorEpredResult', 'PosteriorPredictResult',
    'PosteriorLinpredResult', 'LogLikResult', 'GenericResult',

    'IDLinpred',
    'IDEpred',
    'IDFit',
    'IDLogLik',
    'IDPredict',
    'PriorSpec',

    # stan
    'make_stancode',
    
    # Runtime API
    '_runtime',
    'system_fingerprint',

    "__version__",
]
"""
