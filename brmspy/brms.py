"""
Main brms module with Pythonic API.
"""

import platform
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
from brmspy.brms_functions.io import get_brms_data, read_rds_fit, read_rds_raw, save_rds, get_data
from brmspy.brms_functions.prior import prior, get_prior, default_prior
from brmspy.brms_functions.brm import brm
from brmspy.brms_functions.brm import brm as fit

from brmspy.brms_functions.diagnostics import summary, fixef, ranef, posterior_summary, prior_summary, validate_newdata
from brmspy.brms_functions.generic import call
from brmspy.brms_functions.formula import (
    bf, lf, nlf, acformula, set_rescor, set_mecor, set_nl, 
)
from brmspy.brms_functions.formula import bf as formula
from brmspy.brms_functions.prediction import posterior_epred, posterior_linpred, posterior_predict, log_lik
from brmspy.brms_functions.stan import make_stancode
from brmspy.brms_functions import families
from brmspy.brms_functions.families import (
    brmsfamily, family, student, bernoulli, beta_binomial, negbinomial,
    negbinomial2, geometric, discrete_weibull, com_poisson, lognormal,
    shifted_lognormal, skew_normal, exponential, weibull, frechet,
    gen_extreme_value, exgaussian, wiener, Beta, xbeta, dirichlet,
    dirichlet2, logistic_normal, von_mises, asym_laplace,
    zero_inflated_asym_laplace, cox, hurdle_poisson, hurdle_negbinomial,
    hurdle_gamma, hurdle_lognormal, hurdle_cumulative, zero_inflated_beta,
    zero_one_inflated_beta, zero_inflated_poisson, zero_inflated_negbinomial,
    zero_inflated_binomial, zero_inflated_beta_binomial, categorical,
    multinomial, dirichlet_multinomial, cumulative, sratio, cratio, acat,
    gaussian, poisson, binomial, Gamma, inverse_gaussian
)
from brmspy.runtime._r_packages import install_package as install_rpackage

from brmspy.runtime import (
    install_brms, get_brms_version, install_runtime, deactivate_runtime, activate_runtime,
    get_active_runtime, find_local_runtime
)

# Auto-load last runtime on import
runtime._autoload()

    

# R imports must NOT be done lazily!
# Lazy imports with rpy2 within tqdm loops for example WILL cause segfaults!
# This can lead to wild and unexpected behaviour, hence we do R imports when brms.py is imported

try:
    _get_brms()
except ImportError:
    log_warning("brmspy: brms and other required libraries are not installed. Please call brmspy.install_brms()")


__all__ = [
    # R env
    'install_brms', 'install_runtime', 'get_brms_version',  'deactivate_runtime', 'activate_runtime',
    'find_local_runtime', 'get_active_runtime',

    'install_rpackage',

    # IO
    'get_brms_data', 'save_rds', 'read_rds_raw', 'read_rds_fit', 'get_data',

    # brm
    'fit', 'brm',

    # formula
    'formula', 'bf', 'set_mecor', 'set_rescor', 'set_nl',
    'lf', 'nlf', 'acformula',

    # priors
    'prior', 'get_prior', 'default_prior',

    # prediction
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",

    # diagnosis
    'summary', 'fixef', 'ranef', 'posterior_summary', 'prior_summary', 'validate_newdata',
    
    # generic
    'call',

    # families
    "brmsfamily", "family", "student", "bernoulli", "beta_binomial", "negbinomial",
    "negbinomial2", "geometric", "discrete_weibull", "com_poisson", "lognormal",
    "shifted_lognormal", "skew_normal", "exponential", "weibull", "frechet",
    "gen_extreme_value", "exgaussian", "wiener", "Beta", "xbeta", "dirichlet",
    "dirichlet2", "logistic_normal", "von_mises", "asym_laplace",
    "zero_inflated_asym_laplace", "cox", "hurdle_poisson", "hurdle_negbinomial",
    "hurdle_gamma", "hurdle_lognormal", "hurdle_cumulative", "zero_inflated_beta",
    "zero_one_inflated_beta", "zero_inflated_poisson", "zero_inflated_negbinomial",
    "zero_inflated_binomial", "zero_inflated_beta_binomial", "categorical",
    "multinomial", "dirichlet_multinomial", "cumulative", "sratio", "cratio", "acat",
    "gaussian", "poisson", "binomial", "Gamma", "inverse_gaussian",

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
