"""
Main brms module with Pythonic API.
"""

from brmspy import runtime
from brmspy.helpers.log import log_warning
from brmspy.runtime._state import get_brms as _get_brms
from brmspy.types import (
    FitResult, FormulaResult, GenericResult, LogLikResult, LooResult, LooCompareResult, PosteriorEpredResult, PosteriorLinpredResult, PosteriorPredictResult,
    RListVectorExtension,
    IDLinpred,
    IDEpred,
    IDFit,
    IDLogLik,
    IDPredict,
    PriorSpec
)
from brmspy.brms_functions.io import get_brms_data, read_rds_fit, read_rds_raw, save_rds
from brmspy.brms_functions.prior import prior, get_prior, default_prior
from brmspy.brms_functions.brm import brm, fit
from brmspy.brms_functions.diagnostics import summary, fixef, ranef, posterior_summary, prior_summary, loo, loo_compare, validate_newdata
from brmspy.brms_functions.generic import call
from brmspy.brms_functions.formula import formula
from brmspy.brms_functions.prediction import posterior_epred, posterior_linpred, posterior_predict, log_lik
from brmspy.brms_functions.stan import make_stancode
from brmspy.brms_functions import families
from brmspy.brms_functions.families import family, brmsfamily


# Auto-load last runtime on import
runtime._autoload()

# R imports must NOT be done lazily!
# Lazy imports with rpy2 within tqdm loops for example WILL cause segfaults!
# This can lead to wild and unexpected behaviour, hence we do R imports when brms.py is imported

try:
    _get_brms()
except ImportError:
    log_warning("brmspy: brms and other required libraries are not installed. Please call brmspy.install_brms()")


def install_brms(
    brms_version: str = "latest",
    install_cmdstanr: bool = True,
    install_rstan: bool = False,
    cmdstanr_version: str = "latest",
    rstan_version: str = "latest",
    use_prebuilt_binaries: bool = False,
    install_rtools: bool = False
):
    """
    Install brms R package, optionally cmdstanr and CmdStan compiler, or rstan.
    
    Parameters
    ----------
    brms_version : str, default="latest"
        brms version: "latest", "2.23.0", or ">= 2.20.0"
    install_cmdstanr : bool, default=True
        Whether to install cmdstanr and build CmdStan compiler
    install_rstan : bool, default=False
        Whether to install rstan (alternative to cmdstanr)
    cmdstanr_version : str, default="latest"
        cmdstanr version: "latest", "0.8.1", or ">= 0.8.0"
    rstan_version : str, default="latest"
        rstan version: "latest", "2.32.6", or ">= 2.32.0"
    use_prebuilt_binaries: bool, default=False
        Uses fully prebuilt binaries for cmdstanr and brms and their dependencies.
    install_rtools: bool, default=False
        Installs RTools (windows only) if they cant be found.
    """
    runtime.install(
        use_prebuilt=use_prebuilt_binaries,
        install_rtools=install_rtools,
        brms_version=brms_version,
        cmdstanr_version=cmdstanr_version,
        install_rstan=install_rstan,
        activate=True,
    )


def install_prebuilt(install_rtools: bool = False):
    """Install prebuilt brmspy runtime bundle."""
    return runtime.install(
        use_prebuilt=True,
        install_rtools=install_rtools,
        activate=True,
    )


def get_brms_version() -> str | None:
    """Get installed brms version."""
    return runtime.status().brms_version


def deactivate_runtime():
    """Deactivate current runtime."""
    runtime.deactivate()


def get_active_runtime():
    """Get active runtime path."""
    return runtime.status().active_runtime


__all__ = [
    # R env
    'install_brms', 'get_brms_version', 'install_prebuilt', 'deactivate_runtime', 'get_active_runtime',

    # IO
    'get_brms_data', 'save_rds', 'read_rds_raw', 'read_rds_fit',

    # brm
    'fit', 'brm',

    # formula
    'formula', 

    # priors
    'prior', 'get_prior', 'default_prior',

    # prediction
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",

    # diagnosis
    'summary', 'fixef', 'ranef', 'posterior_summary', 'prior_summary', 'loo', 'loo_compare', 'validate_newdata',
    
    # generic
    'call',

    # families
    'families', 'family', 'brmsfamily',

    # types
    'FitResult', 'FormulaResult', 'PosteriorEpredResult', 'PosteriorPredictResult',
    'PosteriorLinpredResult', 'LogLikResult', 'LooResult', 'LooCompareResult', 'GenericResult', 'RListVectorExtension',

    'IDLinpred',
    'IDEpred',
    'IDFit',
    'IDLogLik',
    'IDPredict',
    'PriorSpec',

    # stan
    'make_stancode'
]
