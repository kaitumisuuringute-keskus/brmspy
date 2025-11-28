"""
brmspy - Python interface to R's brms for Bayesian regression modeling

Provides Pythonic access to the brms R package with proper parameter naming
and seamless arviz integration. Uses brms with cmdstanr backend.

Example:
    >>> from brmspy import brms
    >>> import arviz as az
    >>>
    >>> # First time setup
    >>> brms.install_brms()
    >>>
    >>> # Fit model
    >>> epilepsy = brms.get_brms_data("epilepsy")
    >>> model = brms.fit(
    ...     formula="count ~ zAge + zBase * Trt + (1|patient)",
    ...     data=epilepsy,
    ...     family="poisson",
    ...     chains=4
    ... )
    >>>
    >>> # Analyze
    >>> az.summary(model.idata)
"""

__version__ = "0.1.5"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"

# Import brms module for use as: from brmspy import brms
from brmspy import brms
from brmspy.brms import (
    get_brms_data,
    fit,
    formula,
    install_brms,
    get_brms_version,
    make_stancode,
    posterior_epred,
    posterior_predict,
    posterior_linpred,
    log_lik,
    summary,
    prior,

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
    PriorSpec
)
__all__ = [
    "get_brms_data",
    "fit",
    "formula",
    "install_brms",
    "posterior_predict",
    "posterior_epred",
    "posterior_linpred",
    "log_lik",
    "get_brms_version",
    "make_stancode",
    "summary",
    "prior",

    # return types
    "FitResult",
    "PosteriorEpredResult",
    "PosteriorPredictResult",
    "PosteriorLinpredResult",
    "LogLikResult",
    "GenericResult",
    "FormulaResult",

    # InferenceData types
    "IDLinpred",
    "IDEpred",
    "IDFit",
    "IDLogLik",
    "IDPredict",
    
    "__version__",
]