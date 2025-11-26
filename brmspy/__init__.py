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

__version__ = "0.1.2"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"

# Import brms module for use as: from brmspy import brms
from brmspy import brms
from brmspy.brms import (
    get_brms_data,
    fit,
    install_brms,
    get_brms_version,
    posterior_epred,
    posterior_predict,
    posterior_linpred,
    log_lik,
    FitResult,
    PosteriorEpredResult,
    PosteriorPredictResult,
    GenericResult,
    summary
)
__all__ = [
    "get_brms_data",
    "fit",
    "install_brms",
    "posterior_predict",
    "posterior_epred",
    "posterior_linpred",
    "log_lik",
    "get_brms_version",
    "FitResult",
    "PosteriorEpredResult",
    "PosteriorPredictResult",
    "GenericResult",
    "summary",
    "__version__",
]