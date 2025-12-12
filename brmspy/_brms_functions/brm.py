from collections.abc import Callable, Sequence
from typing import Any, cast

import pandas as pd
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface_lib import openrlib

from brmspy.helpers.log import log, log_warning
from brmspy.types.session_types import SexpWrapper

from ..helpers._rpy2._conversion import brmsfit_to_idata, kwargs_r, py_to_r
from ..helpers._rpy2._priors import _build_priors
from ..types.brms_results import FitResult, IDFit, PriorSpec, ProxyListSexpVector
from ..types.formula_dsl import FormulaConstruct
from .formula import _execute_formula, bf

_formula_fn = bf


_WARNING_CORES = """`cores <= 1` is unsafe in embedded R sessions. The single-process
code path used by brmsfit manipulation or creation functions can crash the interpreter.
Always use `cores >= 2` to force parallel workers and avoid segfaults."""


def _warn_cores(cores: int | None):
    if cores is None or cores <= 1:
        log_warning(_WARNING_CORES)


def brm(
    formula: FormulaConstruct | ProxyListSexpVector | str,
    data: dict | pd.DataFrame,
    priors: Sequence[PriorSpec] | None = None,
    family: str | ListSexpVector | None = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    formula_args: dict | None = None,
    cores: int | None = 2,
    **brm_args,
) -> FitResult:
    """
    Fit Bayesian regression model using brms.

    Uses brms with cmdstanr backend for proper parameter naming.
    Returns FitResult with .idata (arviz.InferenceData) and .r (brmsfit) attributes.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/brm.html)

    Parameters
    ----------
    formula : str
        brms formula: formula string, e.g "y ~ x + (1|group)" or FormulaResult from formula()
    data : dict or pd.DataFrame
        Model data
    priors : list, default=[]
        Prior specifications: [("normal(0,1)", "b"), ("cauchy(0,2)", "sd")]
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", etc.
    sample_prior : str, default="no"
        Sample from prior: "no", "yes", "only"
    sample : bool, default=True
        Whether to sample. If False, returns compiled model with empty=TRUE
    backend : str, default="cmdstanr"
        Stan backend: "cmdstanr" (recommended), "rstan"
    **brm_args
        Additional brms::brm() arguments:
        chains=4, iter=2000, warmup=1000, cores=4, seed=123, thin=1, etc.

    Returns
    -------
    FitResult
        Object with .idata (arviz.InferenceData) and .r (brmsfit) attributes

    See Also
    --------
    brms::brm : R documentation
        https://paulbuerkner.com/brms/reference/brm.html
    posterior_epred : Expected value predictions
    posterior_predict : Posterior predictive samples
    formula : Create formula object with options

    Warnings
    --------
    ``cores <= 1`` is unsafe in embedded R sessions. The single-process
    code path used by ``brms::brm()`` can crash the interpreter.
    Always use ``cores >= 2`` to force parallel workers and avoid segfaults.

    Examples
    --------
    Basic Poisson regression:

    ```python
    from brmspy import brms
    import arviz as az

    epilepsy = brms.get_brms_data("epilepsy")
    model = brms.fit(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        chains=4,
        iter=2000
    )

    az.summary(model.idata)
    ```
    With custom priors:

    ```python
    from brmspy import prior

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
    Survival model with censoring:

    ```python
    kidney = brms.get_brms_data("kidney")

    survival_model = brms.fit(
        formula="time | cens(censored) ~ age + sex + disease + (1|patient)",
        data=kidney,
        family="weibull",
        chains=4,
        iter=4000,
        warmup=2000,
        cores=4,
        seed=42
    )
    ```
    Gaussian model with distributional regression:

    ```python
        # Model both mean and variance
        model = brms.fit(
            formula=brms.formula(
                "y ~ x",
                sigma ~ "z"  # Model heteroscedasticity
            ),
            data=data,
            family="gaussian",
            chains=4
        )
    ```
    """
    import rpy2.robjects as ro
    import rpy2.robjects.packages as packages

    fun_brm = cast(Callable, ro.r("brms::brm"))

    if backend == "cmdstanr":
        try:
            cmdstanr = packages.importr("cmdstanr")
        except:
            cmdstanr = None
        if cmdstanr is None:
            raise RuntimeError(
                "cmdstanr backend is not installed! Please run install_brms(install_cmdstanr=True)"
            )

    if backend == "rstan":
        try:
            rstan = packages.importr("rstan")
        except:
            rstan = None
        if rstan is None:
            raise RuntimeError(
                "rstan backend is not installed! Please run install_brms(install_rstan=True)"
            )

    # Formula checks. These should never be reached in the first place
    # if they are, the library is calling brm() from main directly without remote call
    assert not isinstance(formula, SexpWrapper)
    assert formula is not None
    formula_obj = _execute_formula(formula)

    # Convert data to R format
    data_r = py_to_r(data)

    # Setup priors
    brms_prior = _build_priors(priors)

    # Prepare brm() arguments
    brm_kwargs: dict[str, Any] = {
        "formula": formula_obj,
        "data": data_r,
        "family": family,
        "sample_prior": sample_prior,
        "backend": backend,
        "cores": cores,
    }

    # Add priors if specified
    if len(brms_prior) > 0:
        brm_kwargs["prior"] = brms_prior

    # Add user-specified arguments
    brm_kwargs.update(brm_args)

    brm_kwargs = kwargs_r(brm_kwargs)

    # Set empty=TRUE if not sampling
    if not sample:
        brm_kwargs["empty"] = True
        log("Creating empty r object (no sampling)...")
    else:
        log(f"Fitting model with brms (backend: {backend})...")

    # Call brms::brm() with all arguments
    fit = fun_brm(**brm_kwargs)

    log("Fit done!")

    # Handle return type conversion
    if not sample:
        return FitResult(idata=IDFit(), r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)
