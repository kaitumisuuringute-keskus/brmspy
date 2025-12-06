from typing import Union, Sequence, Optional, List, Tuple, cast
import pandas as pd

from .formula import bf
from brmspy.helpers.log import log
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


_formula_fn = bf



def brm(
    formula: Union[FormulaResult, str],
    data: Union[dict, pd.DataFrame],
    priors: Optional[Sequence[PriorSpec]] = None,
    family: Union[str, ListVector] = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    formula_args: Optional[dict] = None,
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
    }
    
    # Add priors if specified
    if len(brms_prior) > 0:
        brm_kwargs['prior'] = brms_prior
    
    # Add user-specified arguments
    brm_args = kwargs_r(brm_args)
    brm_kwargs.update(brm_args)
    
    # Set empty=TRUE if not sampling
    if not sample:
        brm_kwargs['empty'] = True
        log("Creating empty r object (no sampling)...")
    else:
        log(f"Fitting model with brms (backend: {backend})...")
    
    # Call brms::brm() with all arguments
    fit = brms.brm(**brm_kwargs)
    
    # Handle return type conversion
    if not sample:
        return FitResult(idata=IDFit(), r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)


def add_criterion(
    fit: Union[ListVector, FitResult],
    criterion: Union[str, List[str], Tuple[str]],
    model_name: Optional[str] = None,
    overwrite: Optional[bool] = False,
    file: Optional[str] = None,
    force_save: Optional[bool] = False,
    **kwargs
) -> FitResult:
    """
    Add model fit criteria (e.g. LOO, WAIC) to a brms model.

    Thin wrapper around ``brms::add_criterion()`` that keeps the
    Python-side :class:`FitResult` in sync with the underlying
    ``brmsfit`` object.

    Parameters
    ----------
    fit :
        Fitted model returned by :func:`brm` (``FitResult``) or a raw
        ``brmsfit`` R object (``rpy2.robjects.ListVector``).
    criterion :
        Name or sequence of names of criteria to compute. Supported
        values in brms include ``"loo"``, ``"waic"``, ``"kfold"``,
        ``"loo_subsample"``, ``"bayes_R2"``, ``"loo_R2"``,
        ``"marglik"``.
    model_name :
        Optional model name. If ``None``, brms will derive it from
        the model call.
    overwrite :
        If ``True``, recompute and overwrite criteria that are already
        stored on the model object.
    file :
        Optional base filename for saving the updated model via
        ``saveRDS()`` on the R side. The ``.rds`` extension is added
        automatically. Only used if not ``None``.
    force_save :
        If ``True`` and ``file`` is given, force saving the model even
        if no new criteria were added.
    **kwargs :
        Additional arguments forwarded to the underlying criterion
        functions in brms (for example, arguments to ``loo()``).

    Returns
    -------
    FitResult
        Updated fit with the same posterior draws, but with the
        requested criteria added to the underlying ``brmsfit`` object
        (accessible as ``.r``). If ``fit`` was a :class:`FitResult`,
        its existing ``idata`` is reused. If ``fit`` was a raw
        ``brmsfit``, a new :class:`IDFit` is constructed via
        :func:`brmsfit_to_idata`.

    See Also
    --------
    brms::add_criterion
    brms::loo
    brms::waic
    """
    brms = get_brms()

    # Extract underlying brmsfit and any existing InferenceData
    if isinstance(fit, FitResult):
        fit_r = cast(ListVector, fit.r)
        existing_idata = fit.idata
    else:
        fit_r = cast(ListVector, fit)
        existing_idata = None

    r_kwargs = kwargs_r({
        "criterion": criterion,
        "model_name": model_name,
        "overwrite": overwrite,
        "file": file,
        "force_save": force_save,
        **kwargs,
    })

    fit_r_updated = brms.add_criterion(fit_r, **r_kwargs)

    # Criteria don’t change the posterior draws; reuse idata if we have it
    if isinstance(fit, FitResult) and not isinstance(fit.idata, IDFit):
        # if someone passed a weird FitResult, fall back to recompute
        idata = brmsfit_to_idata(fit_r_updated)
    elif isinstance(fit, FitResult):
        idata = existing_idata
    else:
        idata = brmsfit_to_idata(fit_r_updated)

    if idata is None:
        idata = IDFit()

    return FitResult(idata=idata, r=fit_r_updated)