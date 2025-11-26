"""
brmspy - Pythonic interface to the brms R package using CmdStanPy

A Python package that provides a Pythonic interface to the brms R package,
enabling Bayesian regression modeling through Stan.

Version 0.1 introduces:
- CmdStanPy backend (replaces PyStan)
- Explicit brms version control
- Python 3.10+ support
- Modern packaging standards

Example:
    >>> import brmspy
    >>> # First time setup - install brms
    >>> brmspy.install_brms()  # or install_brms(version="2.23.0")
    >>> 
    >>> # Load example data
    >>> epilepsy = brmspy.get_brms_data("epilepsy")
    >>> 
    >>> # Fit a model
    >>> model = brmspy.fit(
    ...     formula="count ~ zAge + zBase * Trt + (1|patient)",
    ...     data=epilepsy,
    ...     family="poisson"
    ... )
    >>> 
    >>> # View results
    >>> print(model.summary())
"""

__version__ = "0.1.0"
__author__ = "Remi Sebastian Kits, Adam Haber"
__license__ = "Apache-2.0"

# Import main functions from brmspy module
from brmspy.brms import (
    get_brms_data,
    fit,
    install_brms,
    get_brms_version,
    posterior_epred,
    posterior_predict,
    FitResult,
    PosteriorEpredResult,
    PosteriorPredictResult
)

__all__ = [
    "get_brms_data",
    "fit",
    "install_brms",
    "posterior_predict",
    "posterior_epred",
    "get_brms_version",
    "FitResult",
    "PosteriorEpredResult",
    "PosteriorPredictResult",
    "__version__",
]
