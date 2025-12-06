from typing import Union, Sequence, Optional
import pandas as pd


from .formula import bf
from brmspy.helpers.log import log, log_warning
from ..helpers.priors import _build_priors
from ..runtime._state import get_brms, get_cmdstanr, get_rstan
from ..helpers.conversion import (
    brmsfit_to_idata,
    kwargs_r, py_to_r
)
from ..types import (
    FitResult, FormulaResult, IDFit, PriorSpec
)
from rpy2.robjects import ListVector
from rpy2.rinterface_lib import openrlib


_formula_fn = bf


_WARNING_CORES = """`cores <= 1` is unsafe in embedded R sessions. The single-process
code path used by brmsfit manipulation or creation functions can crash the interpreter.
Always use `cores >= 2` to force parallel workers and avoid segfaults."""

def _warn_cores(cores: Optional[int]):
    if cores is None or cores <= 1:
        log_warning(_WARNING_CORES)

def brm(
    formula: Union[FormulaResult, str],
    data: Union[dict, pd.DataFrame],
    priors: Optional[Sequence[PriorSpec]] = None,
    family: Optional[Union[str, ListVector]] = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    formula_args: Optional[dict] = None,
    cores: Optional[int] = 2,
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
    brms = get_brms()

    if backend == "cmdstanr":
        cmdstanr = get_cmdstanr()
        if cmdstanr is None:
            raise RuntimeError("cmdstanr backend is not installed! Please run install_brms(install_cmdstanr=True)")

    if backend == "rstan":
        rstan = get_rstan()
        if rstan is None:
            raise RuntimeError("rstan backend is not installed! Please run install_brms(install_rstan=True)")
    
    
    # Convert formula to brms formula object
    if isinstance(formula, FormulaResult):
        formula_obj = formula.r
    else:
        if formula_args is None:
            formula_args = {}
        formula_obj = _formula_fn(formula, **formula_args).r
    
    # Convert data to R format
    data_r = py_to_r(data)

    # Setup priors
    brms_prior = _build_priors(priors)

    # Prepare brm() arguments
    brm_kwargs = {
        'formula': formula_obj,
        'data': data_r,
        'family': family,
        'sample_prior': sample_prior,
        'backend': backend,
        'cores': cores
    }
    
    # Add priors if specified
    if len(brms_prior) > 0:
        brm_kwargs['prior'] = brms_prior
    
    # Add user-specified arguments
    brm_kwargs.update(brm_args)

    brm_kwargs = kwargs_r(brm_kwargs)
    
    # Set empty=TRUE if not sampling
    if not sample:
        brm_kwargs['empty'] = True
        log("Creating empty r object (no sampling)...")
    else:
        log(f"Fitting model with brms (backend: {backend})...")
    
    # Call brms::brm() with all arguments
    with openrlib.rlock:
        fit = brms.brm(**brm_kwargs)
    
    # Handle return type conversion
    if not sample:
        return FitResult(idata=IDFit(), r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)











