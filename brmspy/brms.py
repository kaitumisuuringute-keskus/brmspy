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
    'fit',
    "posterior_predict", "posterior_epred",
    'FitResult', 'PosteriorEpredResult'
]



def install_brms(version: str = "latest", repo: str = "https://cran.rstudio.com", install_cmdstan: bool = True):
    """
    Install the brms R package, cmdstanr, and CmdStan compiler.
    
    This function provides explicit control over brms and CmdStan installation,
    replacing the automatic installation behavior from previous versions.
    
    Parameters
    ----------
    version : str, default="latest"
        Version specification for brms:
        - "latest": Install the latest version from CRAN
        - Specific version: e.g., "2.23.0"
        - Version constraint: e.g., ">=2.20.0" (requires remotes package)
    
    repo : str, default="https://cran.rstudio.com"
        CRAN repository URL to use for installation
    
    install_cmdstan : bool, default=True
        Whether to also install cmdstanr and CmdStan compiler if not present
    
    Examples
    --------
    Install latest version (brms + cmdstanr + CmdStan):
    >>> import brmspy
    >>> brmspy.install_brms()
    
    Install specific brms version:
    >>> brmspy.install_brms(version="2.23.0")
    
    Install brms only (skip cmdstanr/CmdStan):
    >>> brmspy.install_brms(install_cmdstan=False)
    
    Notes
    -----
    This is a one-time setup operation. Once installed, brms, cmdstanr, and
    CmdStan will be available for all subsequent brmspy sessions.
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
    Get the installed version of the brms R package.
    
    Returns
    -------
    str
        Version string of installed brms package
    
    Raises
    ------
    ImportError
        If brms is not installed
    
    Examples
    --------
    >>> import brmspy
    >>> version = brmspy.get_brms_version()
    >>> print(f"brms version: {version}")
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
    Import example datasets included in the brms package.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to import (e.g., 'epilepsy', 'kidney', 'inhaler')
    
    Returns
    -------
    pd.DataFrame
        The requested dataset as a pandas DataFrame
    
    Examples
    --------
    >>> import brmspy
    >>> epilepsy = brmspy.get_brms_data("epilepsy")
    >>> print(epilepsy.head())
    
    Available datasets include:
    - epilepsy: Epilepsy seizure counts
    - kidney: Kidney infection data  
    - inhaler: Asthma inhaler data
    - And many more - see brms documentation
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
    Generate Stan code for the model using brms.
    
    This calls brms::make_stancode() to generate the Stan program.
    
    Parameters
    ----------
    formula : str
        brms formula specification
    data : dict or pd.DataFrame
        Model data
    priors : list
        List of prior specifications
    family : str
        Distribution family
    sample_prior : str, default="no"
        Whether to sample from prior
    
    Returns
    -------
    str
        Generated Stan code
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
    Fit a Bayesian regression model using brms.
    
    This function uses brms' native fitting capabilities with the cmdstanr
    backend, ensuring proper parameter naming and full brms functionality.
    
    Parameters
    ----------
    formula : str
        Model formula in brms syntax, e.g., "y ~ x + (1|group)"
    data : dict or pd.DataFrame
        Data for the model
    priors : list, optional
        List of prior specifications as tuples, e.g.,
        [("normal(0,1)", "b"), ("cauchy(0,2)", "sd")]
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", etc.
    sample_prior : str, default="no"
        Whether to sample from the prior: "no", "yes", or "only"
    sample : bool, default=True
        Whether to sample from the model. If False, returns compiled model only
        with empty=TRUE.
    backend : str, default="cmdstanr"
        Stan backend to use: "cmdstanr" (recommended), "rstan", or "mock"
    **brm_args
        Additional arguments passed to brms::brm(), e.g.,
        chains=4, iter=2000, warmup=1000, cores=4, seed=123
        
        Common brm() parameters:
        - iter: Total iterations per chain (default: 2000)
        - warmup: Warmup iterations (default: iter/2)
        - chains: Number of chains (default: 4)
        - cores: Number of cores for parallel chains (default: 1)
        - thin: Thinning rate (default: 1)
        - control: Stan control parameters (default: NULL)
        - seed: Random seed (default: NA)
        - silent: Verbosity level (default: 1)
    
    Returns
    -------
    FitResult object with .idata and .r attributes
          allowing access to both Python and R functionality
    
    Examples
    --------
    Basic usage with arviz (default):
    >>> import brmspy
    >>> import arviz as az
    >>> epilepsy = brmspy.get_brms_data("epilepsy")
    >>>
    >>> # Returns arviz InferenceData by default
    >>> idata = brmspy.fit(
    ...     formula="count ~ zAge + zBase * Trt + (1|patient)",
    ...     data=epilepsy,
    ...     family="poisson",
    ...     chains=4,
    ...     iter=2000
    ... )
    >>>
    >>> # Use arviz for analysis
    >>> az.plot_posterior(idata)
    >>> az.summary(idata)
    
    With priors:
    >>> fit = brmspy.fit(
    ...     formula="count ~ zAge + zBase * Trt + (1|patient)",
    ...     data=epilepsy,
    ...     family="poisson",
    ...     priors=[("normal(0, 0.5)", "b"), ("cauchy(0, 1)", "sd")],
    ...     chains=4,
    ...     iter=2000
    ... )
    
    Notes
    -----
    The backend uses brms with cmdstanr, ensuring proper parameter naming
    and full brms functionality. This replaced the direct CmdStanPy approach
    from earlier versions.
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
    brms = _get_brms()
    m = model.r
    data_r = _convert_python_to_R(newdata)

    epred_args = {
        "model": m,
        "newdata": data_r
    }
    epred_args.update(kwargs)

    r = brms.posterior_epred(**epred_args)
    idata = brms_epred_to_idata(r, model.r, newdata=newdata)

    return PosteriorEpredResult(
        r=r, idata=idata
    )

def posterior_predict(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorPredictResult:
    brms = _get_brms()
    m = model.r
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
    else:
        data_r = None

    epred_args = {
        "model": m,
        "newdata": data_r
    }
    epred_args.update(kwargs)

    r = brms.posterior_predict(**epred_args)
    idata = brms_predict_to_idata(r, model.r, newdata=newdata)

    return PosteriorPredictResult(
        r=r, idata=idata
    )

def posterior_linpred(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> GenericResult:
    brms = _get_brms()
    m = model.r
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
    else:
        data_r = None

    epred_args = {
        "model": m,
        "newdata": data_r
    }
    epred_args.update(kwargs)

    r = brms.posterior_linpred(**epred_args)
    idata = brms_linpred_to_idata(r, model.r, newdata=newdata)

    return GenericResult(
        r=r, idata=idata
    )


def log_lik(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> GenericResult:
    brms = _get_brms()
    m = model.r
    if newdata is not None:
        data_r = _convert_python_to_R(newdata)
    else:
        data_r = None

    epred_args = {
        "model": m,
        "newdata": data_r
    }
    epred_args.update(kwargs)

    r = brms.posterior_linpred(**epred_args)
    idata = brms_log_lik_to_idata(r, model.r, newdata=newdata)

    return GenericResult(
        r=r, idata=idata
    )