import typing
import pandas as pd
import numpy as np
import re
import warnings

import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri, numpy2ri, ListVector, DataFrame, StrVector
from rpy2.robjects.conversion import localconverter
from .helpers.priors import _build_priors
from .helpers.singleton import _get_base, _get_brms, _get_cmdstanr, _get_rstan, _invalidate_singletons
from .helpers.conversion import (
    brmsfit_to_idata,
    brms_linpred_to_idata, brms_log_lik_to_idata, brms_epred_to_idata, brms_predict_to_idata,
    kwargs_r, py_to_r,
    r_to_py
)
from .types import (
    FitResult, FormulaResult, GenericResult, IDEpred, IDFit, IDLinpred, IDLogLik,
    IDPredict, LogLikResult, PosteriorEpredResult, PosteriorLinpredResult, PosteriorPredictResult, PriorSpec,
    prior
)
from .install import install_brms

# R imports must NOT be done lazily! 
# Lazy imports with rpy2 within tqdm loops for example WILL cause segfaults!
# This can lead to wild and unexpected behaviour, hence we do R imports when brms.py is imported
try:
    _get_brms()
except ImportError:
    print("brmspy: brms and other required libraries are not installed. Please call brmspy.install_brms()")

print("brmspy <0.2 is still evolving; APIs may change.")
print("Feedback or a star on GitHub helps guide development:")
print("https://github.com/kaitumisuuringute-keskus/brmspy")

__all__ = [
    'install_brms', 'get_brms_version', 'get_brms_data', 'make_stancode',
    'fit', 'formula', 'summary',
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",
    'FitResult', 'FormulaResult', 'PosteriorEpredResult', 'PosteriorPredictResult',
    'PosteriorLinpredResult', 'LogLikResult', 'GenericResult',
    "prior"
]

def get_brms_version() -> str:
    """
    Get installed brms R package version.
    
    Returns
    -------
    str
        Version string (e.g., "2.23.0")
    
    Raises
    ------
    ImportError
        If brms is not installed
    
    Examples
    --------

    ```python
    from brmspy import brms
    version = brms.get_brms_version()
    print(f"brms version: {version}")
    ```
    """
    brms = _get_brms()
    utils = rpackages.importr("utils")
    
    # Get package version
    version_info = utils.packageVersion("brms")
    # Convert to string and clean up R output format
    version_str = str(version_info[0]).strip()
    
    # R returns format like '[1]  2 22  0' - extract just the numbers
    # and format as proper version string
    import re
    numbers = re.findall(r'\d+', version_str)
    if len(numbers) >= 3:
        return '.'.join(numbers[:3])
    elif len(numbers) == 2:
        return '.'.join(numbers)
    else:
        # Fallback - return cleaned string
        return version_str.replace('[1]', '').strip()




def get_brms_data(dataset_name: str) -> pd.DataFrame:
    """
    Load example dataset from brms package.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name. Available datasets include:
        - 'epilepsy': Epileptic seizure counts
        - 'kidney': Kidney infection data with censoring
        - 'inhaler': Inhaler usage study
        - 'btdata': British Telecom share price data
        - And many more from brms package
    
    Returns
    -------
    pd.DataFrame
        Dataset as pandas DataFrame with column names preserved
    
    See Also
    --------
    brms::brmsdata : R documentation for available datasets
        https://paulbuerkner.com/brms/reference/index.html#data
    
    Examples
    --------
    Load epilepsy dataset:
    
    ```python
    from brmspy import brms
    epilepsy = brms.get_brms_data("epilepsy")
    print(epilepsy.head())
    print(epilepsy.columns)
    ```

    Load kidney dataset with censoring:
    
    ```python
    kidney = brms.get_brms_data("kidney")
    print(f"Shape: {kidney.shape}")
    print(f"Censored observations: {kidney['censored'].sum()}")
    ```
    """
    brms = _get_brms()
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        brmsdata = rpackages.data(brms)
        if brmsdata:
            return pd.DataFrame(brmsdata.fetch(dataset_name)[dataset_name])
        else:
            return pd.DataFrame({})




def make_stancode(
    formula: typing.Union[FormulaResult, str],
    data: pd.DataFrame,
    priors: typing.Optional[typing.Sequence[PriorSpec]] = None,
    family: str = "poisson",
    sample_prior: str = "no",
    formula_args: typing.Optional[dict] = None
) -> str:
    """
    Generate Stan code using brms::make_stancode().
    
    Useful for inspecting the generated Stan model before fitting,
    understanding the model structure, or using the code with other
    Stan interfaces.
    
    Parameters
    ----------
    formula : str or FormulaResult
        brms formula specification
    data : pd.DataFrame
        Model data
    priors : list of PriorSpec, optional
        Prior specifications from prior() function
    family : str, default="poisson"
        Distribution family (gaussian, poisson, binomial, etc.)
    sample_prior : str, default="no"
        Whether to sample from prior:
        - "no": No prior samples
        - "yes": Include prior samples alongside posterior
        - "only": Sample from prior only (no data)
    formula_args : dict, optional
        Additional arguments passed to formula()
    
    Returns
    -------
    str
        Complete Stan program code as string
    
    See Also
    --------
    brms::make_stancode : R documentation
        https://paulbuerkner.com/brms/reference/make_stancode.html
    fit : Fit model instead of just generating code
    make_standata : Generate Stan data block
    
    Examples
    --------
    Generate Stan code for simple model:
    
    ```python
    from brmspy import brms
    epilepsy = brms.get_brms_data("epilepsy")
    
    stan_code = brms.make_stancode(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson"
    )
    
    print(stan_code[:500])  # Print first 500 characters
    ```

    With custom priors:
    
    ```python
        from brmspy import prior
        
        stan_code = brms.make_stancode(
            formula="count ~ zAge",
            data=epilepsy,
            priors=[prior("normal(0, 1)", class_="b")],
            family="poisson"
        )
    ```

    For prior predictive checks (sample_prior="only"):

    ```
    stan_code = brms.make_stancode(
        formula="count ~ zAge",
        data=epilepsy,
        family="poisson",
        sample_prior="only"
    )
    ```
    """
    brms = _get_brms()

    data_r = py_to_r(data)
    priors_r = _build_priors(priors)
    if isinstance(formula, FormulaResult):
        formula_obj = formula.r
    else:
        if formula_args is None:
            formula_args = {}
        formula_obj = _formula_fn(formula, **formula_args).r


    if len(priors_r) > 0:
        return brms.make_stancode(
            formula=formula_obj, data=data_r, prior=priors_r, family=family, sample_prior=sample_prior
        )[0]
    else:
        return brms.make_stancode(
            formula=formula_obj, data=data_r, family=family, sample_prior=sample_prior
        )[0]




def formula(
    formula: str,
    **formula_args
) -> FormulaResult:
    """
    Set up a model formula for brms package.
    
    Allows defining (potentially non-linear) additive multilevel models
    for all parameters of the assumed response distribution.

    Parameters
    ----------
    formula : str
        brms formula specification, e.g., "y ~ x + (1|group)"
    **formula_args : dict
        Additional brms::brmsformula() arguments:
        
        - decomp : str
            Decomposition method (e.g., "QR" for QR decomposition)
        - center : bool
            Whether to center predictors (default True)
        - sparse : bool
            Use sparse matrix representation
        - nl : bool
            Whether formula is non-linear
        - loop : bool
            Use loop-based Stan code
    
    Returns
    -------
    FormulaResult
        Object with .r (R brmsformula object) and .dict (Python dict) attributes
    
    See Also
    --------
    brms::brmsformula : R documentation
        https://paulbuerkner.com/brms/reference/brmsformula.html
    fit : Fit model using formula
    
    Examples
    --------
    Basic formula:
    
    ```python
        from brmspy import brms
        f = brms.formula("y ~ x1 + x2 + (1|group)")
    ```

    With QR decomposition for numerical stability:
    
    ```python
    f = brms.formula(
        "reaction ~ days + (days|subject)",
        decomp="QR"
    )
    ```

    Multivariate formula:
    ```
        f = brms.formula(
            "mvbind(y1, y2) ~ x1 + x2",
            center=True
        )
    ```
    """
    brms = _get_brms()
    formula_args = kwargs_r(formula_args)
    formula_obj = brms.bf(formula, **formula_args)
    return FormulaResult(r=formula_obj, dict=typing.cast(dict, r_to_py(formula_obj)))

_formula_fn = formula


def fit(
    formula: typing.Union[FormulaResult, str],
    data: typing.Union[dict, pd.DataFrame],
    priors: typing.Optional[typing.Sequence[PriorSpec]] = None,
    family: str = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    formula_args: typing.Optional[dict] = None,
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
    brms = _get_brms()

    if backend == "cmdstanr":
        cmdstanr = _get_cmdstanr()
        if cmdstanr is None:
            raise RuntimeError("cmdstanr backend is not installed! Please run install_brms(install_cmdstanr=True)")

    if backend == "rstan":
        rstan = _get_rstan()
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
        print("Creating empty r object (no sampling)...")
    else:
        print(f"Fitting model with brms (backend: {backend})...")
    
    # Call brms::brm() with all arguments
    fit = brms.brm(**brm_kwargs)
    
    # Handle return type conversion
    if not sample:
        return FitResult(idata=IDFit(), r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)

def posterior_epred(model: FitResult, newdata: pd.DataFrame, **kwargs) -> PosteriorEpredResult:
    """
    Compute expected value of posterior predictive distribution.
    
    Calls brms::posterior_epred() to get E[Y|data] without observation noise.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/posterior_epred.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame
        Data for predictions
    **kwargs
        Additional arguments to brms::posterior_epred()
    
    Returns
    -------
    PosteriorEpredResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_posterior_epred = typing.cast(typing.Callable, ro.r('brms::posterior_epred'))
    
    # Call with proper argument names (object instead of model)
    r = r_posterior_epred(m, newdata=data_r, **kwargs)
    idata = brms_epred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDEpred, idata)

    return PosteriorEpredResult(
        r=r, idata=idata
    )

def posterior_predict(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorPredictResult:
    """
    Generate posterior predictive samples with observation noise.
    
    Calls brms::posterior_predict() to get samples of Y_new|data.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs
        Additional arguments to brms::posterior_predict()
    
    Returns
    -------
    PosteriorPredictResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_posterior_predict = typing.cast(typing.Callable, ro.r('brms::posterior_predict'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_predict(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_predict(m, **kwargs)
    
    idata = brms_predict_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDPredict, idata)

    return PosteriorPredictResult(
        r=r, idata=idata
    )

def posterior_linpred(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorLinpredResult:
    """
    Compute linear predictor of the model.
    
    Returns samples of the linear predictor (before applying the link function).
    Useful for understanding the model's predictions on the linear scale.

    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs : dict
        Additional arguments to brms::posterior_linpred():
        
        - transform : bool - Apply inverse link function (default False)
        - ndraws : int - Number of posterior draws
        - summary : bool - Return summary statistics
    
    Returns
    -------
    PosteriorLinpredResult
        Object with .idata (IDLinpred) and .r (R matrix) attributes
    
    See Also
    --------
    brms::posterior_linpred : R documentation
        https://paulbuerkner.com/brms/reference/posterior_linpred.brmsfit.html
    posterior_epred : Expected values on response scale
    
    Examples
    --------
    ```python
        from brmspy import brms
        
        epilepsy = brms.get_brms_data("epilepsy")
        model = brms.fit(
            "count ~ zAge + zBase * Trt + (1|patient)",
            data=epilepsy,
            family="poisson",
            chains=4
        )
        
        # Linear predictor (log scale for Poisson)
        linpred = brms.posterior_linpred(model)
        print(linpred.idata.predictions)
    ```
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_posterior_linpred = typing.cast(typing.Callable, ro.r('brms::posterior_linpred'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_linpred(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_linpred(m, **kwargs)
    
    idata = brms_linpred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLinpred, idata)

    return PosteriorLinpredResult(
        r=r, idata=idata
    )


def log_lik(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> LogLikResult:
    """
    Compute log-likelihood values.
    
    Returns log p(y|theta) for each observation. Essential for model
    comparison via LOO-CV and WAIC.

    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs : dict
        Additional arguments to brms::log_lik():
        
        - ndraws : int - Number of posterior draws
        - combine_chains : bool - Combine chains (default True)
    
    Returns
    -------
    LogLikResult
        Object with .idata (IDLogLik) and .r (R matrix) attributes
    
    See Also
    --------
    brms::log_lik : R documentation
        https://paulbuerkner.com/brms/reference/log_lik.brmsfit.html
    arviz.loo : Leave-One-Out Cross-Validation
    arviz.waic : Widely Applicable Information Criterion
    
    Examples
    --------
    Compute log-likelihood for model comparison:
    
    ```python
    from brmspy import brms
    import arviz as az
    
    epilepsy = brms.get_brms_data("epilepsy")
    model = brms.fit(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        chains=4
    )
    
    # LOO-CV for model comparison
    loo = az.loo(model.idata)
    print(loo)
    ```

    Compare multiple models:
    ```python
    model1 = brms.fit("count ~ zAge + (1|patient)", data=epilepsy, family="poisson", chains=4)
    model2 = brms.fit("count ~ zAge + zBase + (1|patient)", data=epilepsy, family="poisson", chains=4)
    
    comp = az.compare({'model1': model1.idata, 'model2': model2.idata})
    print(comp)
    ```
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_log_lik = typing.cast(typing.Callable, ro.r('brms::log_lik'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_log_lik(m, newdata=data_r, **kwargs)
    else:
        r = r_log_lik(m, **kwargs)
    
    idata = brms_log_lik_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLogLik, idata)

    return LogLikResult(
        r=r, idata=idata
    )


def summary(model: FitResult, **kwargs) -> pd.DataFrame:
    """
    Generate summary statistics for fitted model.
    
    Calls R's summary() function on brmsfit object and converts to pandas DataFrame.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    **kwargs
        Additional arguments to summary(), e.g., probs=c(0.025, 0.975)
    
    Returns
    -------
    pd.DataFrame
        Summary statistics with columns for Estimate, Est.Error, and credible intervals
    
    See Also
    --------
    brms::summary.brmsfit : R documentation
        https://paulbuerkner.com/brms/reference/summary.brmsfit.html
    
    Examples
    --------
    ```python
    from brmspy import brms
    
    model = brms.fit("y ~ x", data=data, chains=4)
    summary_df = brms.summary(model)
    print(summary_df)
    ```
    """
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    
    kwargs = kwargs_r(kwargs)

    # Get R summary function
    r_summary = typing.cast(typing.Callable, ro.r('summary'))
    
    # Call summary on brmsfit object
    summary_r = r_summary(model.r, **kwargs)
    
    # Extract the fixed effects table (summary$fixed)
    # brms summary returns a list with $fixed, $random, $spec_pars, etc.
    try:
        fixed_table = typing.cast(typing.Callable, ro.r('function(x) as.data.frame(x$fixed)'))(summary_r)
        
        # Convert to pandas
        with localconverter(default_converter + pandas2ri.converter):
            summary_df = pandas2ri.rpy2py(fixed_table)
        
        return summary_df
    except Exception:
        # Fallback: just convert the whole summary to string
        summary_str = str(summary_r)
        print(summary_str)
        return pd.DataFrame({'summary': [summary_str]})