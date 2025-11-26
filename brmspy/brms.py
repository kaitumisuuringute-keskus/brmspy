import typing
import pandas as pd
import numpy as np
import re
import warnings

import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri, numpy2ri, ListVector, DataFrame, StrVector
from rpy2.robjects.conversion import localconverter
from .helpers import (
    _get_brms, _convert_python_to_R, 
    brmsfit_to_idata,
    brms_linpred_to_idata, brms_log_lik_to_idata, brms_epred_to_idata, brms_predict_to_idata
)
from .types import (
    FitResult, GenericResult, PosteriorEpredResult, PosteriorPredictResult
)

__all__ = [
    'install_brms', 'get_brms_version', 'get_brms_data', 'get_stan_code',
    'fit', 'summary',
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",
    'FitResult', 'PosteriorEpredResult'
]



def install_brms(version: str = "latest", repo: str = "https://cran.rstudio.com", install_cmdstan: bool = True):
    """
    Install brms R package, cmdstanr, and CmdStan compiler.
    
    Parameters
    ----------
    version : str, default="latest"
        brms version: "latest", "2.23.0", or ">=2.20.0"
    repo : str, default="https://cran.rstudio.com"
        CRAN repository URL
    install_cmdstan : bool, default=True
        Whether to install cmdstanr and CmdStan
    
    Examples
    --------
    >>> from brmspy import brms
    >>> brms.install_brms()
    >>> brms.install_brms(version="2.23.0")
    >>> brms.install_brms(install_cmdstan=False)
    """
    print("=" * 60)
    print("brmspy Setup - Installing Required Components")
    print("=" * 60)
    
    # Install cmdstanr and CmdStan if requested
    if install_cmdstan:
        print("\n[1/3] Installing cmdstanr R package...")
        try:
            cmdstanr = rpackages.importr("cmdstanr")
            print("✓ cmdstanr already installed")
        except:
            print("Installing cmdstanr from r-universe...")
            utils = rpackages.importr("utils")
            utils.install_packages(
                StrVector(('cmdstanr',)),
                repos=StrVector(('https://stan-dev.r-universe.dev', repo))
            )
            print("✓ cmdstanr installed successfully!")
        
        print("\n[2/3] Checking CmdStan installation...")
        try:
            import rpy2.robjects as ro
            ro.r('library(cmdstanr); cmdstan_path()')
            print("✓ CmdStan already installed")
        except Exception:
            print("Installing CmdStan compiler via cmdstanr...")
            print("This may take several minutes depending on your system.")
            try:
                import rpy2.robjects as ro
                ro.r('library(cmdstanr); install_cmdstan()')
                print("✓ CmdStan installed successfully!")
            except Exception as e:
                warnings.warn(f"Could not install CmdStan: {e}")
                print("You can install it manually later in R with:")
                print("  library(cmdstanr)")
                print("  install_cmdstan()")
    
    # Install brms
    print(f"\n[3/3] Installing brms R package (version: {version})...")
    
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    
    if version == "latest":
        # Install latest version from CRAN
        print("Installing latest brms from CRAN...")
        utils.install_packages(StrVector(('brms',)), repos=repo)
        print("✓ brms installed successfully!")
    else:
        # Install specific version using remotes package
        print(f"Installing brms version {version}...")
        try:
            remotes = rpackages.importr("remotes")
        except:
            print("Installing remotes package (required for version-specific installation)...")
            utils.install_packages(StrVector(('remotes',)))
            remotes = rpackages.importr("remotes")
        
        # Handle version constraints
        if ">=" in version or "<=" in version or "==" in version:
            # Version constraint - install latest matching version
            warnings.warn(
                f"Version constraint '{version}' specified. "
                "Installing latest version that matches. "
                "For reproducibility, specify exact version (e.g., '2.23.0')"
            )
            # Extract version number for comparison
            version_num = version.replace(">=", "").replace("<=", "").replace("==", "").strip()
            remotes.install_version("brms", version=version_num, repos=repo)
        else:
            # Exact version
            remotes.install_version("brms", version=version, repos=repo)
        
        print(f"✓ brms {version} installed successfully!")
    
    # Clear cached brms import to force reload
    global _brms
    _brms = None
    
    # Verify installation
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    if install_cmdstan:
        try:
            import rpy2.robjects as ro
            cmdstan_path = str(ro.r('library(cmdstanr); cmdstan_path()')[0])
            print(f"✓ cmdstanr: installed")
            print(f"✓ CmdStan: {cmdstan_path}")
        except:
            print("⚠ cmdstanr/CmdStan: Not installed")
    
    try:
        brms_version = get_brms_version()
        print(f"✓ brms: version {brms_version}")
    except Exception as e:
        print(f"⚠ brms: Could not verify ({e})")
    
    print("\n✅ Setup complete! You're ready to use brmspy.")
    print("\nExample usage:")
    print("  import brmspy")
    print("  epilepsy = brmspy.get_brms_data('epilepsy')")
    print("  fit = brmspy.fit('count ~ zAge + (1|patient)', epilepsy, family='poisson', chains=4)")


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
        Dataset name ('epilepsy', 'kidney', 'inhaler', etc.)
    
    Returns
    -------
    pd.DataFrame
        Dataset as pandas DataFrame
    
    Examples
    --------
    >>> from brmspy import brms
    >>> epilepsy = brms.get_brms_data("epilepsy")
    """
    brms = _get_brms()
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        return pd.DataFrame(rpackages.data(brms).fetch(dataset_name)[dataset_name])




def get_stan_code(
    formula: str,
    data: typing.Union[dict, pd.DataFrame],
    priors: list,
    family: str,
    sample_prior: str = "no"
) -> str:
    """
    Generate Stan code using brms::make_stancode().
    
    Parameters
    ----------
    formula : str
        brms formula
    data : dict or pd.DataFrame
        Model data
    priors : list
        Prior specifications
    family : str
        Distribution family
    sample_prior : str, default="no"
        Sample from prior ("no", "yes", "only")
    
    Returns
    -------
    str
        Stan program code
    """
    brms = _get_brms()
    if len(priors) > 0:
        return brms.make_stancode(
            formula=formula, data=data, prior=priors, family=family, sample_prior=sample_prior
        )[0]
    else:
        return brms.make_stancode(
            formula=formula, data=data, family=family, sample_prior=sample_prior
        )[0]






def fit(
    formula: str,
    data: typing.Union[dict, pd.DataFrame],
    priors: list = [],
    family: str = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    **brm_args,
) -> FitResult:
    """
    Fit Bayesian regression model using brms.
    
    Uses brms with cmdstanr backend for proper parameter naming.
    Returns FitResult with .idata (arviz.InferenceData) and .r (brmsfit) attributes.
    
    Parameters
    ----------
    formula : str
        brms formula: "y ~ x + (1|group)"
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
    
    Examples
    --------
    >>> from brmspy import brms
    >>> import arviz as az
    >>>
    >>> epilepsy = brms.get_brms_data("epilepsy")
    >>> model = brms.fit(
    ...     formula="count ~ zAge + zBase * Trt + (1|patient)",
    ...     data=epilepsy,
    ...     family="poisson",
    ...     chains=4,
    ...     iter=2000
    ... )
    >>> az.summary(model.idata)
    """
    brms = _get_brms()
    
    # Check if cmdstanr backend is available
    if backend == "cmdstanr":
        try:
            import rpy2.robjects.packages as rpackages
            rpackages.importr("cmdstanr")
        except:
            raise ImportError(
                "cmdstanr R package not found. Install it using:\n\n"
                "  install.packages('cmdstanr', repos = c('https://stan-dev.r-universe.dev', getOption('repos')))\n\n"
                "Or use backend='rstan' (requires rstan package)"
            )
    
    # Convert formula to brms formula object
    formula_obj = brms.bf(formula)
    
    # Convert data to R format
    data_r = _convert_python_to_R(data)

    # Setup priors
    if len(priors) > 0:
        brms_prior = brms.prior_string(*priors[0])
        for p in priors[1:]:
            brms_prior = brms_prior + brms.prior_string(*p)
        assert brms.is_brmsprior(brms_prior)
    else:
        brms_prior = []

    # Prepare brm() arguments
    brm_kwargs = {
        'formula': formula_obj,
        'data': data_r,
        'family': family,
        'sample_prior': sample_prior,
        'backend': backend,
    }
    
    # Add priors if specified
    if len(priors) > 0:
        brm_kwargs['prior'] = brms_prior
    
    # Add user-specified arguments
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
        return FitResult(idata=[], r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)

def posterior_epred(model: FitResult, newdata: pd.DataFrame, **kwargs) -> PosteriorEpredResult:
    """
    Compute expected value of posterior predictive distribution.
    
    Calls brms::posterior_epred() to get E[Y|data] without observation noise.
    
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
    data_r = _convert_python_to_R(newdata)

    # Get R function explicitly
    r_posterior_epred = ro.r('brms::posterior_epred')
    
    # Call with proper argument names (object instead of model)
    r = r_posterior_epred(m, newdata=data_r, **kwargs)
    idata = brms_epred_to_idata(r, model.r, newdata=newdata)

    return PosteriorEpredResult(
        r=r, idata=idata
    )

def posterior_predict(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorPredictResult:
    """
    Generate posterior predictive samples with observation noise.
    
    Calls brms::posterior_predict() to get samples of Y_new|data.
    
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
    
    # Get R function explicitly
    r_posterior_predict = ro.r('brms::posterior_predict')
    
    # Call with proper arguments
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
        r = r_posterior_predict(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_predict(m, **kwargs)
    
    idata = brms_predict_to_idata(r, model.r, newdata=newdata)

    return PosteriorPredictResult(
        r=r, idata=idata
    )

def posterior_linpred(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> GenericResult:
    """
    Compute linear predictor of the model.
    
    Calls brms::posterior_linpred() to get samples of the linear predictor.
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs
        Additional arguments to brms::posterior_linpred()
    
    Returns
    -------
    GenericResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r
    
    # Get R function explicitly
    r_posterior_linpred = ro.r('brms::posterior_linpred')
    
    # Call with proper arguments
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
        r = r_posterior_linpred(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_linpred(m, **kwargs)
    
    idata = brms_linpred_to_idata(r, model.r, newdata=newdata)

    return GenericResult(
        r=r, idata=idata
    )


def log_lik(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> GenericResult:
    """
    Compute log-likelihood values.
    
    Calls brms::log_lik() to get log p(y|theta) for each observation.
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs
        Additional arguments to brms::log_lik()
    
    Returns
    -------
    GenericResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    _get_brms()  # Ensure brms is loaded
    m = model.r
    
    # Get R function explicitly
    r_log_lik = ro.r('brms::log_lik')
    
    # Call with proper arguments
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
        r = r_log_lik(m, newdata=data_r, **kwargs)
    else:
        r = r_log_lik(m, **kwargs)
    
    idata = brms_log_lik_to_idata(r, model.r, newdata=newdata)

    return GenericResult(
        r=r, idata=idata
    )


def summary(model: FitResult, **kwargs) -> pd.DataFrame:
    """
    Generate summary statistics for fitted model.
    
    Calls R's summary() function on brmsfit object and converts to pandas DataFrame.
    
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
    
    Examples
    --------
    >>> from brmspy import brms
    >>> model = brms.fit("y ~ x", data=data, chains=4)
    >>> summary_df = brms.summary(model)
    >>> print(summary_df)
    """
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    
    # Get R summary function
    r_summary = ro.r('summary')
    
    # Call summary on brmsfit object
    summary_r = r_summary(model.r, **kwargs)
    
    # Extract the fixed effects table (summary$fixed)
    # brms summary returns a list with $fixed, $random, $spec_pars, etc.
    try:
        fixed_table = ro.r('function(x) as.data.frame(x$fixed)')(summary_r)
        
        # Convert to pandas
        with localconverter(default_converter + pandas2ri.converter):
            summary_df = pandas2ri.rpy2py(fixed_table)
        
        return summary_df
    except Exception:
        # Fallback: just convert the whole summary to string
        summary_str = str(summary_r)
        print(summary_str)
        return pd.DataFrame({'summary': [summary_str]})