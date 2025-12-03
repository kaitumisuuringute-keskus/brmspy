
from brmspy.binaries.use import autoload_last_runtime
from brmspy.helpers.log import log_warning
from brmspy.helpers.singleton import _get_brms
from brmspy.types import (
    FitResult, FormulaResult, GenericResult, LogLikResult, PosteriorEpredResult, PosteriorLinpredResult, PosteriorPredictResult,
    IDLinpred,
    IDEpred,
    IDFit,
    IDLogLik,
    IDPredict,
    PriorSpec
)
from brmspy.install import install_brms, install_prebuilt, get_brms_version
from brmspy.brms_functions.io import get_brms_data, read_rds_fit, read_rds_raw, save_rds
from brmspy.brms_functions.prior import prior, get_prior, default_prior
from brmspy.brms_functions.brm import brm, fit
from brmspy.brms_functions.diagnosis import summary
from brmspy.brms_functions.formula import formula
from brmspy.brms_functions.prediction import posterior_epred, posterior_linpred, posterior_predict, log_lik
from brmspy.brms_functions.stan import make_stancode
from brmspy.brms_functions import families
from brmspy.brms_functions.families import family, brmsfamily

autoload_last_runtime()

# R imports must NOT be done lazily!
# Lazy imports with rpy2 within tqdm loops for example WILL cause segfaults!
# This can lead to wild and unexpected behaviour, hence we do R imports when brms.py is imported

try:
    _get_brms()
except ImportError:
    log_warning("brmspy: brms and other required libraries are not installed. Please call brmspy.install_brms()")

__all__ = [
    # R env
    'install_brms', 'get_brms_version', 'install_prebuilt',

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
    'summary',

    # families
    'families', 'family', 'brmsfamily',

    # types
    'FitResult', 'FormulaResult', 'PosteriorEpredResult', 'PosteriorPredictResult',
    'PosteriorLinpredResult', 'LogLikResult', 'GenericResult',

    'IDLinpred',
    'IDEpred',
    'IDFit',
    'IDLogLik',
    'IDPredict',
    'PriorSpec',

    # stan
    'make_stancode'
]



